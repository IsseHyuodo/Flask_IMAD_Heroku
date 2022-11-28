from flask import Flask,request,jsonify
import pickle
import numpy as np

model=pickle.load(open('RandomForest.pkl','rb'))
app=Flask(__name__)
@app.route('/')
def home():
    return "Hello World"
   

@app.route('/predict',methods=['POST','GET'])
def predict():
    if request.method == 'POST':
         location=request.form.get('Location')
         personal_info=request.form.get('Personal info') 
         financial_info=request.form.get('Financial info')
         health_and_fitness=request.form.get('Health and Fitness')
         messages=request.form.get('Messages')
         photos_and_videos=request.form.get('Photos and videos') 
         audio=request.form.get('Audio')
         files_and_docs=request.form.get('Files and docs')
         contacts=request.form.get('Contacts')
         app_activity=request.form.get('App activity') 
         app_info_and_performance=request.form.get('App info and performance')
         device_or_other_ids=request.form.get('Device or other IDs')
  
         input_query=np.array([[location,personal_info,financial_info,
                    health_and_fitness,messages,photos_and_videos,
                    audio,files_and_docs,contacts,app_activity,app_info_and_performance,
                    device_or_other_ids]])
         result1=model.predict(input_query)
         output=result1[0]
         return jsonify({'Genre':str(output)})
   
    else:

        return "Hello"
  


if __name__=='__main__':
    app.run(debug=True)






