import React, { useState, useEffect } from 'react';
import './App.css';
import { PDFViewer, PdfFocusProvider } from '@llamaindex/pdf-viewer';

function App() {
  const [text,changetext] = useState('');
  const [output,setout] = useState('');
  const [context,setcontext] = useState('');
  const [documenturl,seturl] = useState('');
  const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

  const  Handle = async ()=>
    {
            if(questions[text]===undefined){
              Newchat();
              setout("Enter a valid text");
            }
            else{
            setout(output+
                  <div className='qa'>
                      <div className='question'>
                       <p style={{'fontSize':'34px'}}>⚪</p>
                       <p style={{'fontSize':'22px'}}>{text}</p>
                      </div>
                      <div className="bubble"></div>
                    </div>
            );
            setcontext();
            await sleep(3000);
            //pdf - fetch 
            var urls_pdf = {};
            for (let i = 0; i < 5; i++){
              const pdf = await fetch(`http://localhost:5000/fetch-pdf?url=${questions[text].sources['url'][i]}`);
              console.log(pdf);
              const pdfBlob = await pdf.blob();
              const pdfUrl = URL.createObjectURL(pdfBlob);
              urls_pdf[i]=pdfUrl
            }
            
            console.log(urls_pdf);
            setout(
              <div className='qa'>
                  <div className='question'>
                   <p style={{'fontSize':'34px'}}>⚪</p>
                   <p style={{'fontSize':'22px'}}>{text}</p>
                  </div>
                  {/* <div className="bubble"></div> */}
                  <div className='answer'>
                    <p style={{'fontSize':'28px'}}>🟢</p>
                    <p style={{'fontSize':'18px'}}>{questions[text].answer}</p>
                    </div>
                </div>
        );
            setcontext(
                    <div className='context'>
                    <div className='contextbox' onClick={(e) => seturl( 
                      <iframe 
                        src={urls_pdf[0]}
                        width="100%" 
                        height="100%"
                        title="PDF Viewer" >
                      </iframe>
                    )}>
                      {questions[text].sources[0].slice(0,130)}...
                    </div>
                    <div  className='contextbox' onClick={(e) => seturl( 
                      <iframe 
                        src={urls_pdf[1]}
                        width="100%" 
                        height="100%"
                        title="PDF Viewer" >
                      </iframe>
                    )}>
                      {questions[text].sources[1].slice(0,130)}...
                    </div>
                    <div  className='contextbox' onClick={(e) => seturl( 
                      <iframe 
                        src={urls_pdf[2]}
                        width="100%" 
                        height="100%"
                        title="PDF Viewer" >
                      </iframe>
                    )}>
                      {questions[text].sources[2].slice(0,130)}...
                    </div>
                    <div  className='contextbox' onClick={(e) => seturl( 
                      <iframe 
                        src={urls_pdf[3]}
                        width="100%" 
                        height="100%"
                        title="PDF Viewer" >
                      </iframe>
                    )}>
                      {questions[text].sources[3].slice(0,130)}...
                    </div>
                    <div  className='contextbox' onClick={(e) => seturl( 
                      <iframe 
                        src={urls_pdf[4]}
                        width="100%" 
                        height="100%"
                        title="PDF Viewer" >
                      </iframe>
                    )}>
                      {questions[text].sources[4].slice(0,130)}...
                    </div>
                  </div>
                )
  }
};
  //NEW CHAT BUTTON => RESET STATES
  const Newchat = ()=>{
    changetext('');
    setout('');
    setcontext('');
    seturl();
  };

  const [data, setData] = useState([]);
  useEffect(() => {
          fetch('final_response_log.json')
            .then(response => response.json())
            .then(jsonData => {
              setData(jsonData);
            })
            .catch(error => console.error('Error fetching data:', error));
        }, []);

  var questions ={};
  for (let i = 0; i < data.length; i++) {
    if(data[i].sources.length === 0)
    {continue; }
    else{
    questions[data[i].question] = {'answer':data[i].answer,'sources':{'0':data[i].sources[0].message,
      '1':data[i].sources[1].message,
      '2':data[i].sources[2].message,
      '3':data[i].sources[3].message,
      '4':data[i].sources[4].message,
      'url':{'0':data[i].sources[0].source_url,'1':data[i].sources[1].source_url,'2':data[i].sources[2].source_url,'3':data[i].sources[3].source_url,'4':data[i].sources[4].source_url}
    }};

  }
  }
  console.log(questions);

  
  return (
    <div className="MainContainer"> 
                
                <div className='Page'>
                    <div className='Header1'>
                      <div className='chatbutton'>
                      <button onClick={Newchat} className='newchat'>      
                        <span className="plus-icon">+</span>  New Chat
                      </button>
                      </div>
                      <h3 className='contextheading'>Context:</h3>
                        {context}
                    </div>
                    <div className='Header2'>
                      <div className='Message'>

                      </div>
                      <div className='Input'>
                        <div className='Content'>
                          {output}                          
                          </div>
                        <div className='button-text'>
                        <input type="text" placeholder='Enter text:'  value={text} onInput={(e) => changetext(e.target.value)} required />
                        <button onClick={Handle}>➤</button>
                        </div>
                      </div>
                    </div>
                    <div className='Header3'>
                      <div style={{'height':'100vh'}}>{documenturl}</div>
                        
                    </div>
                </div>
    
      
    </div>
  );
}

export default App;
