
const express = require("express");
const request = require('request');
const cors = require('cors');

const app = express();

const port = process.env.PORT || 5000;

// Routes & Middleware
app.use(express.json());
app.use(cors());
app.get('/fetch-pdf', (req, res) => {
    const url = req.query.url;
    request({ url, encoding: null }, (err, response, body) => {
      if (err) {
        res.status(500).send('Error occurred while fetching the PDF');
      } else {
        res.set('Content-Type', 'application/pdf');
        res.send(body);
      }
    });
  });
  

app.listen(port, () => {
    console.log(`Server running on port ${port}`);
  });