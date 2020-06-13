# CRGC-NLP
ML Model and Live Website to detect incomplete cancer pathology reports in real-time.

# Project Aim

The aim of this project was to create and deploy a machine learning model via a web application to allow pathologists to submit a bladder cancer report and get real time alerts via email if the submitted report was classified as incomplete. 

The key features required were:
<ol class= "lead">
  <li> A NLP machine learning model to parse and classify a pathology report. This was achieved using a custom vectorizer and multi-model architecture to maximize the utilization of the medical data provided.</li>
  <li> A live web application to deploy the model as a proof of concept for a viable and scalable solution to provide pathologists access to feedback. This was achieved by developing a web application using Flask and deploying on Heroku.</li>
  <li> During EDA and model development, significant keywords were found as key identifiers in the reports that tied well with the CRGC's mission to motivate better structured reports. Therefore, the keywords used are also displayed for each submitted path report.</li>
</ol>

# Team

The following students worked on this project:
<ol class= "lead">
  <li> Peru Dayani</li>
      <ul>
        <li> Developed model architecture and models using python and sckit-learn.
        <li> Developed custom TF-IDF text vectorizer for medical data. 
        <li> Developed web app using Flask, python, HTML, bootstrap and JS.
        <li> Deployed web app on Heroku and maintains it.
        <li> Lead team meetings with CRGC representatives to set realistic goals.
      </ul>
  <li> Carlos Calderon</li>
        <ul>
          <li> Compared ML model architectures 
        </ul>       
  <li> Saumya Choudhary</li>
        <ul>
          <li> Conducted EDA on data
          <li> Developed data cleaning pipeline
        </ul>  
</ol>

# Project Results

Live Website: https://crgc-mvp.herokuapp.com/ (Under development)
Final Deliverable Demo: https://www.youtube.com/embed/gZLGlP98EsA
<br>
Final Deliverable Presentation: https://drive.google.com/open?id=1srR26ON6Vu-ygoowqm9AqW7AJcM6NW0Y03QRXd-njM4

