import streamlit as st
from langchain_core.output_parsers import StrOutputParser
import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import langchain
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_openai import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import TokenTextSplitter
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.summarize import load_summarize_chain
from langchain import PromptTemplate
import pandas as pd
from dotenv import load_dotenv
load_dotenv()


# Prompt Template
prompt_welcome='''Please extract and provide the following details from the earnings call transcript:

Data = {text}

### Example Format:

Company Information:

Company Name: [Company Name]
Quarter: [Quarter]
Type of Call: [Type of Call]
Conference Call Date: [Conference Call Date]
Number of Pages in Transcript: [Number of Pages]
Management Information:

Name	Designation
[Name 1]	[Designation 1]
[Name 2]	[Designation 2]
'''

prompt_speaker_text ='''Please provide all the text spoken by {speaker} in the following transcript. Include all statements, responses, and any other remarks made by this individual.
Data = {text}
#YOU ARE EXPECTED TO OUTPUT ONLY THE TEXT SPOKEN BY THE SPEAKER.
#If YOU cant provide then you are expected to output NIL'''

prompt_topics='''Please provide all the TOPICS SPOKEN in the following transcript. Include all necessary topics, points,details,numericals and cluster them into points of different topics which might comprise the transcript summary into points.Do not exceed max of 30 topics and try to cluster the transcript within the max 30.
Data = {text}
#YOU ARE EXPECTED TO OUTPUT ONLY THE Important Topics(key events or topics discussed) with a brief line in points of 1.,2.,3.....'''

summarize_template_text = '''You're an TOPIC WISE Transcript Summarizer designed to effectively extract the sentence that are related to the topic that is given by the user.
            Your primary function is to extract key information while preserving the context and core messages of the content provided. DONT FORGET TO INCLUDE THE NUMERICALS
            WHICH ARE RELATED TO THE CONTEXT.
            USER CONTEXT = '''
            
summarize_use='''Task: Summarize the following text based on the user's context input.
Text to Summarize: ['{text}']
When summarizing, ensure to highlight key points, essential details, and the overall message of the text while keeping the summary concise and coherent. Maintain the logical flow of information 
and focus on capturing the main ideas without unnecessary details.
Remember, your goal is to Summarize the Transcript data for the user context and provide the output in points(don't include subheading for each point).
OUTPUT Only the necessary points with max of 10points with each one or two lines max.
'''

transcript_label_prompt ='''From the Given Transcript, Label which Sentence is question and which one is answer and also tag the person spoken the sentence.
Data = {text}
#You are expected to provide the output as a table with the speakers full sentence in a column,corresponding indicator representing Q or A,along the speaker of the particular sentence in another column also add S.No (start from 1) ,Company Name: Determine the company the speaker belongs to by analyzing phrases that typically mention the company, such as "from," "representing," or within the introduction.
Management Status: Determine whether the speaker is part of the management team. Consider titles like "CEO," "CFO," "Vice President," "Director," "Head of," or similar executive roles. If such a title is found near the speaker's name or in the context of their introduction, return True; otherwise, return False.
.
If it is neither Q/A mark N/A.
PROVIDE ONLY THE TABLE AS AN OUTPUT in this format
| S.No | Sentence | Q/A | Speaker | Company Name | Management Status |.
'''

# GPT-4o-mini
llm =  ChatOpenAI(model='gpt-4o-mini',temperature=0,max_tokens=None) 

# reading the uploaded pdf using langchain-PyPDFLoader
def read_pdf(doc):
    from langchain.document_loaders import PyPDFLoader
    loader = PyPDFLoader(doc)
    raw_documents = loader.load()
    text_splitter = TokenTextSplitter(chunk_size=500,chunk_overlap=100) 
    documents = text_splitter.split_documents(raw_documents)
    embeddings = OpenAIEmbeddings()
    vectorIndex = FAISS.from_documents(documents, embeddings)  # RAG Embeddings
    return documents, vectorIndex

def table_gen(output_chat,sno_list,sentence_list,qa_list,speaker_list,company_list,status_list):
    lines = output_chat.content.splitlines()
    lines = lines[2:]  # Skip the header and separator lines
    for line in lines:
        if line.strip() == '' or line.startswith('|------'):
            continue
        parts = line.split('|')
        sno = parts[1].strip()
        sentence = parts[2].strip()
        qa = parts[3].strip()
        speaker = parts[4].strip()
        company = parts[5].strip()
        status = parts[6].strip()
        sno_list.append(sno)
        sentence_list.append(sentence)
        qa_list.append(qa)
        speaker_list.append(speaker)
        company_list.append(company)
        status_list.append(status)
    return sno_list,sentence_list,qa_list,speaker_list,company_list,status_list,len(lines)



# define session state variables
if 'pdf_text' not in st.session_state:
    st.session_state.pdf_text = None

if 'vectorIndex' not in st.session_state:
    st.session_state.vectorIndex = None

if 'names' not in st.session_state:
    st.session_state.names=None
    
if 'designations' not in st.session_state:
    st.session_state.designations = None

if 'speaker_out' not in st.session_state:
    st.session_state.speaker_out = []

if 'processed' not in st.session_state:
    st.session_state.processed = False
    
if 'topics_generated' not in st.session_state:
    st.session_state.topics_generated = False
    
if 'local_data' not in st.session_state:
    st.session_state.local_data = []
    
if 'sections' not in st.session_state:
    st.session_state.sections = []
    
if 'questions_generated' not in st.session_state:
    st.session_state.questions_generated = False

if 'questionsdf' not in st.session_state:
    st.session_state.questionsdf = ''
    
if 'tab' not in st.session_state:
    st.session_state.tab = 'Welcome'

if 'cost' not in st.session_state:
    st.session_state.cost = 0

if 'input_tok' not in st.session_state:
    st.session_state.input_tok = 0

if 'output_tok' not in st.session_state:
    st.session_state.output_tok = 0

#GPT-4o-mini Pricing
#$0.150 / 1M input tokens
#$0.600 / 1M output tokens
inp_cost=0.150/1000000
out_cost = 0.600/1000000
# function call to update the st.session_state.cost for each task
def cost_cal(out):
    global inp_cost,out_cost
    st.session_state.input_tok+=out.usage_metadata['input_tokens']
    st.session_state.output_tok+=out.usage_metadata['output_tokens']
    cost = str(st.session_state.input_tok*inp_cost + st.session_state.output_tok*out_cost)
    return cost

# Use columns to display buttons horizontally
# Define buttons to mimic tabs, displayed in rows
if st.sidebar.button("Welcome"):
    st.session_state.tab = 'Welcome'
if st.sidebar.button("Opening remarks summary"):
    st.session_state.tab = 'Opening remarks summary'
if st.sidebar.button("Question answer summary"):
    st.session_state.tab = 'Question answer summary'
if st.sidebar.button("Chatbot"):
    st.session_state.tab = 'Chatbot'
    

# Welcome Tab --------------------------------------------------------------------------------------------------------------------
if st.session_state.tab == "Welcome":
    st.title("Summarize Transcripts and Download")
    st.write("Upload your PDF file:")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        # Read the PDF and store it in session_state
        file_path = "uploaded_file.pdf"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state.pdf_text,st.session_state.vectorIndex = read_pdf(file_path)
        st.success("PDF loaded successfully!")
        but = st.button("Process")
        if(but):
            #resetting the state variables
            st.session_state.cost = 0
            st.session_state.input_tok = 0
            st.session_state.output_tok = 0
            st.session_state.names=None
            st.session_state.designations = None
            st.session_state.speaker_out = []
            st.session_state.processed = False
            st.session_state.topics_generated = False
            st.session_state.local_data = []
            st.session_state.sections = []
            st.session_state.questionsdf = ''
            prompt = ChatPromptTemplate.from_template(prompt_welcome)
            chain = prompt | llm
            output_welcome = chain.invoke({"text":st.session_state.pdf_text})
            st.session_state.cost = cost_cal(output_welcome)
            with st.sidebar:
                st.write("Token counter:", st.session_state.input_tok + st.session_state.output_tok)
                st.write("Cost:", st.session_state.cost)
            st.write(output_welcome.content)
            management_info = re.findall(r'\| ([^\|]+?)\s+\| ([^\|]+?)\s+\|', output_welcome.content)
            if management_info[0][0].strip() == "Name" and management_info[0][1].strip() == "Designation":
                management_info = management_info[1:]
            #names and designations of the speakers
            st.session_state.names = [info[0].strip() for info in management_info]
            st.session_state.designations = [info[1].strip() for info in management_info]        
            st.session_state.processed=False            #resetes the session state after the process
            st.session_state.topics_generated = False
            st.session_state.questions_generated=False
        else:
            with st.sidebar:
                st.write("Token counter:", st.session_state.input_tok + st.session_state.output_tok)
                st.write("Cost:", st.session_state.cost)
    else:
            with st.sidebar:
                st.write("Token counter:", st.session_state.input_tok + st.session_state.output_tok)
                st.write("Cost:", st.session_state.cost)
                
# Opening remarks summary Tab ---------------------------------------------------------------------------------------------
elif st.session_state.tab == "Opening remarks summary":
    st.title("Opening remarks summary")
    st.text("This page will help you summarize opening remarks.")
    st.text("To get started,expand raw text section fr overview")
    st.text("\nOnce ready, Click Get Topics")
    if not st.session_state.processed:
        # Speaker wise content extraction
        local = []
        for i in st.session_state.names:
            prompt = ChatPromptTemplate.from_template(prompt_speaker_text)
            chain = prompt | llm
            output_chat = chain.invoke({"text": st.session_state.pdf_text, "speaker": i})
            st.session_state.speaker_out += [output_chat.content]
            dic = {"name": i, "text": output_chat.content}
            local.append(dic)
            st.session_state.cost = cost_cal(output_chat)
        st.session_state.local_data = local
        st.session_state.processed = True
        with st.sidebar:
                st.write("Token counter :",st.session_state.input_tok+st.session_state.output_tok)
                st.write("Cost : ", st.session_state.cost)  
    with st.expander('Who said what?'):
        st.data_editor(st.session_state.local_data, disabled=True)

    # Get Topics button logic
    but_top = st.button("Get Topics")
    if but_top and not st.session_state.topics_generated:
        prompt = ChatPromptTemplate.from_template(prompt_topics)
        chain = prompt | llm
        output_chat = chain.invoke({"text": ''.join(st.session_state.speaker_out)})
        st.session_state.cost = cost_cal(output_chat)
        pattern = re.compile(r'\d+\.\s*\*\*(.*?)\*\*\s*-?\s*(.*?)(?=\d+\.\s*\*\*|\Z)', re.DOTALL)
        matches = pattern.findall(output_chat.content)
        st.session_state.sections = [f"**{match[0]}**\n- {match[1].strip()}" for match in matches]
        st.session_state.topics_generated = True
        
    if st.session_state.topics_generated:
        with st.expander('Topics'):
            user_top = st.text_input("Enter your choice (e.g., 'all' or topic number):")
            but = st.button("Get topic-wise summary")
            if but:
                countpg=0
                for i in st.session_state.pdf_text:
                    countpg+=len(i.page_content)
                if user_top.lower() == 'all':
                    all_summaries = []
                    for i in st.session_state.sections:
                        prompt = PromptTemplate(
                            input_variables=['text'],
                            template=summarize_template_text + i + summarize_use
                        )
                        chain = load_summarize_chain(
                            llm,
                            chain_type='stuff',
                            prompt=prompt,
                            verbose=False
                        )
                        output_summary = chain.run(st.session_state.pdf_text)
                        st.session_state.input_tok += countpg
                        st.session_state.output_tok += len(output_summary)
                        all_summaries.append(output_summary)
                        

                    st.download_button(
                        label="Download Processed File",
                        data='\n'.join(all_summaries),
                        file_name="processed_file.txt",
                        mime="text/plain",
                    )
                    with st.sidebar:
                        st.write("Token counter :",st.session_state.input_tok+st.session_state.output_tok)
                        st.write("Cost : ", st.session_state.cost)
                    for i,summary in enumerate(st.session_state.sections,all_summaries):
                        st.write(i)
                        st.write(summary)

                elif(int(user_top)<len(st.session_state.sections)+1):
                    
                    selected_section = st.session_state.sections[int(user_top) - 1]

                    prompt = PromptTemplate(
                        input_variables=['text'],
                        template=summarize_template_text + selected_section + summarize_use
                    )
                    chain = load_summarize_chain(
                        llm,
                        chain_type='stuff',
                        prompt=prompt,
                        verbose=False
                    )
                    output_summary = chain.run(st.session_state.pdf_text)
                    st.download_button(
                        label="Download Processed File",
                        data=output_summary,
                        file_name="processed_file.txt",
                        mime="text/plain",
                    )
                    st.write(selected_section)
                    st.write(output_summary)
                    st.session_state.input_tok += countpg
                    st.session_state.output_tok += len(output_summary)
                    st.session_state.cost = str(st.session_state.input_tok * inp_cost + st.session_state.output_tok * out_cost)
                    with st.sidebar:
                        st.write("Token counter :",st.session_state.input_tok+st.session_state.output_tok)
                        st.write("Cost : ", st.session_state.cost)
                else:
                    st.write("Enter valid range")
                    with st.sidebar:
                        st.write("Token counter:", st.session_state.input_tok + st.session_state.output_tok)
                        st.write("Cost:", st.session_state.cost)
            else:
                for i,sec in enumerate(st.session_state.sections):
                    st.write(i+1,sec)
                with st.sidebar:
                    st.write("Token counter:", st.session_state.input_tok + st.session_state.output_tok)
                    st.write("Cost:", st.session_state.cost)
                    
#Questions/Answers Summary -------------------------------------------------------------------------------------------------------
elif st.session_state.tab == "Question answer summary":
    st.title("Questions remarks summary")
    st.text("This page will help you summarize Q&A")
    st.text("To get started, expand raw text section for overview")
    if(not st.session_state.questions_generated):
        with st.expander('Questions'):
            topic_wise_output=[]
            counter = 1
            sno_list = []
            sentence_list = []
            qa_list = []
            speaker_list = []
            company_list = []
            status_list = []
            sn = []
            coun = 1
            for i in st.session_state.pdf_text:
                prompt = ChatPromptTemplate.from_template(transcript_label_prompt)
                chain = prompt | llm
                output_chat = chain.invoke({"text": i,'i':counter})
                topic_wise_output+=[output_chat.content]
                lines = output_chat.content.count("|")
                counter+=(lines-10)/5
                st.session_state.cost = cost_cal(output_chat)
                sno_list,sentence_list,qa_list,speaker_list,company_list,status_list,l=table_gen(output_chat,sno_list,sentence_list,qa_list,speaker_list,company_list,status_list)
                sn+=[coun]*l
                coun+=1
            with st.sidebar:
                st.write("Token counter :",st.session_state.input_tok+st.session_state.output_tok)
                st.write("Cost : ", st.session_state.cost)
            df = pd.DataFrame({
                                "chunk":sn,
                                'Sentence': sentence_list,
                                'Q/A': qa_list,
                                'Speaker': speaker_list,
                                'company':company_list,
                                'status':status_list
                            })
            #post processing
            df['company'].replace('N/A', pd.NA, inplace=True)
            df['company'] = df.groupby('Speaker')['company'].apply(lambda x: x.ffill().bfill())

            df=df[df['Q/A']!='N/A'].reset_index().drop("index",axis=1)
           
            st.data_editor(df,disabled=True)
            st.session_state.questions_generated = True
            st.session_state.questionsdf=df
    else: 
        with st.expander('Expand'):
            st.data_editor(st.session_state.questionsdf,disabled=True)
        with st.sidebar:
            st.write("Token counter:", st.session_state.input_tok + st.session_state.output_tok)
            st.write("Cost:", st.session_state.cost)

#ChatBot Tab--------------------------------------------------------------------------------------------------------------------
elif st.session_state.tab =='Chatbot':
    st.write("Chatbot")
    #RAG
    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=st.session_state.vectorIndex.as_retriever())
    langchain.debug = True
    user_input = st.text_input("Enter question:")
    but_rag = st.button("Ask")
    if(len(user_input) > 0 and but_rag):
        output_chat = chain({"question":user_input})
        st.write(output_chat['answer'])
        st.session_state.input_tok+=len(user_input)
        st.session_state.output_tok+=len(output_chat)
        st.session_state.cost = str(st.session_state.input_tok * inp_cost + st.session_state.output_tok * out_cost)
        with st.sidebar:
            st.write("Token counter:", st.session_state.input_tok + st.session_state.output_tok)
            st.write("Cost:", st.session_state.cost)
    else:
        with st.sidebar:
            st.write("Token counter:", st.session_state.input_tok + st.session_state.output_tok)
            st.write("Cost:", st.session_state.cost)
