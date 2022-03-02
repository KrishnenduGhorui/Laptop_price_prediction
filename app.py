from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Loading trained ML model and Scaling object from pickle 

model=pickle.load(open('model_rfr.pickle','rb'))
dict_Scalers=pickle.load(open('dict_Scalers.pickle','rb'))
dict_LEs=pickle.load(open('dict_LEs.pickle','rb'))
valCountle7_company_lap=pickle.load(open('valCountle7_company_lap.pickle','rb'))
valCounte1_Gpu_type_lap=pickle.load(open('valCounte1_Gpu_type_lap.pickle','rb'))
valCountl10_Cpu_Type_intsty_lap=pickle.load(open('valCountl10_Cpu_Type_intsty_lap.pickle','rb'))

@app.route('/',methods=['GET'])
def Home():
    return render_template('home.html')


@app.route("/predict", methods=['POST'])
def predict():
    # Taking input from front end page 
    if request.method == 'POST':
        Company=str(request.form['Company'])		
        TypeName=str(request.form['TypeName'])
        Inches=float(request.form['Inches'])
        ScreenResolution=str(request.form['ScreenResolution'])		
        Cpu=str(request.form['Cpu'])
        Ram=str(request.form['Ram'])
        Memory=str(request.form['Memory'])
        Gpu=str(request.form['Gpu'])
        OpSys=str(request.form['OpSys'])
        Weight=str(request.form['Weight'])
						
        screenType=ScreenResolution.split(' ')[0]
        if screenType in ['IPS','Full','Touchscreen','Quad','4K']:
            ScreenType=screenType         
        elif screenType=='1366x768':
            ScreenType='Other1'
        else:
            ScreenType='Other2'
    
        SRPxlH=int(ScreenResolution.split(' ')[-1].split('x')[0])
        SRPxlV=int(ScreenResolution.split(' ')[-1].split('x')[1])
    
       	Cpu_Company=Cpu.split(' ')[0]
        Cpu_Type=Cpu.split(' ')[1]
        Cpu_Type_intensity=Cpu.split(' ')[2]
        Cpu_Ghz=float(Cpu.split(' ')[-1][:-3])
        if Cpu_Type_intensity in valCountl10_Cpu_Type_intsty_lap:
            Cpu_Type_intensity='Cpu_Type_intensity_other'
			 
        Ram_GB=int(float(str(Ram).replace('GB','')))
		
        Memory=Memory.replace('Flash Storage','FlashStorage')
        if '+' in Memory:
            Memory1,Memory2=Memory.split(' +  ')      
        else:
            Memory1=Memory
            Memory2=0
		
        Memory1_size,Memory1_type=Memory1.split(' ')
        if Memory2!=0:
            Memory2_size,Memory2_type=Memory2.split(' ')
        else:
            Memory2_size=int(0)
            Memory2_type='NotPresent'
         
        if 'TB' in Memory1_size:
            Memory1_size_GB=int(float(Memory1_size[:-2]))*1000
        else:
            Memory1_size_GB=int(float(Memory1_size[:-2]))
			
        if Memory2_size==0:
            Memory2_size_GB=int(0)
        elif 'TB' in Memory2_size:
            Memory2_size_GB=int(float(Memory2_size[:-2]))*1000
        elif 'GB' in Memory2_size:
            Memory2_size_GB=int(float(Memory2_size[:-2]))
        
        Gpu_company=Gpu.split(' ')[0]
        Gpu_type=Gpu.split(' ')[1]   
        if Gpu_type in valCounte1_Gpu_type_lap:
            Gpu_type='Gpu_type_other' 
	   
        Weight_kg=float(str(Weight).replace('kg',''))
		
		# Encoding 
        Company=dict_LEs['Company'].transform([Company])[0]
        TypeName=dict_LEs['TypeName'].transform([TypeName])[0]
        OpSys=dict_LEs['OpSys'].transform([OpSys])[0]
        Memory1_type=dict_LEs['Memory1_type'].transform([Memory1_type])[0]
        Memory2_type=dict_LEs['Memory2_type'].transform([Memory2_type])[0]
        Cpu_Company=dict_LEs['Cpu_Company'].transform([Cpu_Company])[0]
        Cpu_Type=dict_LEs['Cpu_Type'].transform([Cpu_Type])[0]
        Cpu_Type_intensity=dict_LEs['Cpu_Type_intensity'].transform([Cpu_Type_intensity])[0]
        Gpu_company=dict_LEs['Gpu_company'].transform([Gpu_company])[0]
        Gpu_type=dict_LEs['Gpu_type'].transform([Gpu_type])[0]
        ScreenType=dict_LEs['ScreenType'].transform([ScreenType])[0]
		
		
        # Standard Scaling some features
        #Company=dict_Scalers['Company'].transform(np.array(Company).reshape(1,1))[0][0]
        #TypeName=dict_Scalers['TypeName'].transform(np.array(TypeName).reshape(1,1))[0][0]
		
        X=np.array([Company,TypeName,Inches,Ram_GB,OpSys,Weight_kg,Memory1_size_GB,Memory1_type,Memory2_size_GB,Memory2_type,Cpu_Company,Cpu_Type,Cpu_Type_intensity,Cpu_Ghz,Gpu_company,Gpu_type,ScreenType,SRPxlH,SRPxlV]).reshape(1,-1)
        # Predicting target 
        prediction=model.predict(X)[0]        		
        
        return render_template('home.html',prediction_text="The price of laptop is "+str(round(prediction,2))+' euros')
        
    else:
        return render_template('home.html')

if __name__=="__main__":
    app.run(debug=True)