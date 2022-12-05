import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit as st
from PIL import Image
from streamlit_option_menu import option_menu
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA



forest_fires = pd.read_csv("https://raw.githubusercontent.com/Nitish-Satya-Sai/data-is-crucial/main/forest_fires.csv")

st.markdown(" # 🌳🌳 Predicting the Fire Occurence in the Forest 🌳🌳")


forest_fires=forest_fires.iloc[:,1:]


columns_list = forest_fires.columns.to_list()

numerical_columns = columns_list.copy()

categorical_columns = columns_list.copy()

del numerical_columns[0:3]
del numerical_columns[-1]

del categorical_columns[0:13]
Target = forest_fires.iloc[:,-1]

Target_encoded = Target.replace(to_replace={"not_fire":0,"fire":1})
X = forest_fires.iloc[:,3:-1]
X.columns=["Temperature in Celsius degrees","Relative Humidity in %","Wind speed in km/h",
           "Rain(total day in mm)","Fine Fuel Moisture Code (FFMC)","Duff Moisture Code (DMC)",
           "Drought Code (DC) index","Initial Spread Index (ISI)","Buildup Index (BUI)","Fire Weather Index (FWI)"]

def helper_friend(data_x,data_y,choice,classifier):
    X_train,X_test,y_train,y_test = train_test_split(data_x,data_y,test_size=0.3,random_state=42)
    if (choice==1):
        

        # we are creating an object/instance for a class called StandardScaler.
        my_scaler = StandardScaler()

        # we are fitting the data. Fitting is nothing but calculation of required parameters for scaling
        #standard scaler requires the mean and the standard deviation of the data. 
        # It takes mu and sigma and compute it for each and every instance of a column.

        my_scaler.fit(X_train) # why isn't y_train used here?

        # There is no point of scaling y which is an integer, 
        # it doesn't make any sense of scaling the output.

        # we are transforming the X_train, it is nothing 
        #but computing the above formula for each column in X_train
        X_train_scaled = my_scaler.transform(X_train)
        # we are transforming the X_test, it is nothing 
        #but computing the above formula for each column in X_test
        X_test_scaled = my_scaler.transform(X_test)


        # note that this code doesn't care which estimator you chose

        my_model = classifier.fit(X_train_scaled, y_train)
        predictions = my_model.predict(X_test_scaled)
        Acc = accuracy_score(y_test,predictions)
        cfm = confusion_matrix(y_test,predictions)
        return (Acc,cfm)
        
        
    elif(choice==0):
        # note that this code doesn't care which estimator you chose

        my_model = classifier.fit(X_train, y_train)
        predictions = my_model.predict(X_test)
        Acc = accuracy_score(y_test,predictions)
        cfm = confusion_matrix(y_test,predictions)
        return (Acc,cfm)
def helper_2(X_train, X_test,y_train,y_test,classifier):
    my_model = classifier.fit(X_train,y_train)
    return accuracy_score(y_test,my_model.predict(X_test))
        

        



# 1. as sidebar menu
with st.sidebar:
    st.write("#### 🙌🙌🙌🙌 Feel Free to select 😁.")
    optionm = option_menu("Main menu", ["The Motivation behind the project","Model Building Stack","Forest Fires Prediction"
                                        ,"The Power of PCA",
           "👉 Feel free for mining to discover more insights 😀🤔😁😱😎","Exploratory Data Analysis",], default_index=0)




if optionm=="The Motivation behind the project":
    st.write("# The Motivation behind the project")
    im = Image.open("fires.jpg") 
    st.image(im)
    st.write("[Image Source:](http://www.electronic-sirens.com/use-of-sirens-for-forest-fires/)")
    st.write("##### The main concentric goal🎯 of taking this initiative of doing this work is to predict the chance of forest fires based on various parameters. If we take the statistics from 2012 to 2021, an average of 7.4 million acres of forest are impacted annually due to forest fires 😱. We can at least reduce the impact of these forest fires if we can able to predict the chance of occurrence of fires based on the various parameters. ")
    
    st.write("## Dataset description:")
    st.write('''
             
             
             ##### To reach my goal, I researched the potential datasets, which is vital to begin digging the valuable insights from the data. Moreover, surprisingly, I found one dataset while I was going through the University of California, Irvine, repositories. And that, too, is the latest dataset donated in 2019. 
             
             
             ''')
    st.write('''
             
             
            ##### This dataset aids in achieving my end goal. This dataset contains **14 features** with **244 records/instances**. This dataset includes weather data observations and FWI components. Based on these values, each record/instance is classified as “fire” & “not_fire". The complete form of FWI is Forest Weather Index. The FWI is a system that is used worldwide to estimate fire danger. This is also a well-balanced dataset where 138 instances belong to the “fire” class, and 106 instances belong to the not_fire class.
             
             
             ''')
    st.write('''
             
             
           ## In-depth description of the Dataset
             
             ''')
    st.write('''
             #### About Dataset features

1. Date : (DD/MM/YYYY) Day, month ('june' to 'september'), year (2012)

#### Weather data observations

2. Temp : temperature noon (temperature max) in Celsius degrees: 22 to 42
3. RH : Relative Humidity in %: 21 to 90
4. Ws :Wind speed in km/h: 6 to 29
5. Rain: total day in mm: 0 to 16.8

#### FWI Components

6. Fine Fuel Moisture Code (FFMC) index from the FWI system: 28.6 to 92.5
7. Duff Moisture Code (DMC) index from the FWI system: 1.1 to 65.9
8. Drought Code (DC) index from the FWI system: 7 to 220.4
9. Initial Spread Index (ISI) index from the FWI system: 0 to 18.5
10. Buildup Index (BUI) index from the FWI system: 1.1 to 68
11. Fire Weather Index (FWI) Index: 0 to 31.1
12. Classes: two classes, namely fire and not fire
             ''')



elif optionm=="Model Building Stack":
    
    tab1,tab2,tab3,tab4 = st.tabs(["Info Tab","Scores Evaluation","Validating the models with K fold Cross-Validation technique","HyperParameter Tuning"])
    with tab1:
        st.write("This is where I evaluated the model through a series of steps")
    with tab2:
        options = st.radio("Please select one option",options=["Scaling the Input Features","Without Scaling the Input Features"])


        st.header("K-Nearest Neighbors Classifier")
        if options=="Scaling the Input Features":
            knn = KNeighborsClassifier(n_neighbors=5)
            acc,cfm = helper_friend(X,Target_encoded,1,knn)

            st.metric("Accuracy",str(round(acc,4)*100)+"%")
 
            plt.figure(figsize=(3,2))
            fig1 = sns.heatmap(cfm,annot=True,cmap="rocket_r")
            st.pyplot(plt.gcf())
        elif options=="Without Scaling the Input Features":
            knn = KNeighborsClassifier(n_neighbors=5)
            acc,cfm = helper_friend(X,Target_encoded,0,knn)

            st.metric("Accuracy",str(round(acc,4)*100)+"%")
            plt.figure(figsize=(3,2))
            fig2 = sns.heatmap(cfm,annot=True,cmap="rocket_r")
            st.pyplot(plt.gcf())

        st.header("Decision Tree Classifier 🌳")
        if options=="Scaling the Input Features":
            DTC = DecisionTreeClassifier(random_state=42)
            acc,cfm = helper_friend(X,Target_encoded,1,DTC)

            st.metric("Accuracy",str(round(acc,4)*100)+"%")
            plt.figure(figsize=(3,2))
            fig3 = sns.heatmap(cfm,annot=True,cmap="rocket_r")
            st.pyplot(plt.gcf())
        elif options=="Without Scaling the Input Features":
            DTC = DecisionTreeClassifier(random_state=42)
            acc,cfm = helper_friend(X,Target_encoded,1,DTC)

            st.metric("Accuracy",str(round(acc,4)*100)+"%")
            plt.figure(figsize=(3,2))
            fig4 = sns.heatmap(cfm,annot=True,cmap="rocket_r")
            st.pyplot(plt.gcf())
        

        st.header("Random Forest Classifier 🌳🌳🌳🌳........")
        if options=="Scaling the Input Features":
            RFC = RandomForestClassifier(random_state=42)
            acc,cfm = helper_friend(X,Target_encoded,1,RFC)

            st.metric("Accuracy",str(round(acc,4)*100)+"%")
            plt.figure(figsize=(3,2))
            fig5 = sns.heatmap(cfm,annot=True,cmap="rocket_r")
            st.pyplot(plt.gcf())
        elif options=="Without Scaling the Input Features":
            RFC = RandomForestClassifier(random_state=42)
            acc,cfm = helper_friend(X,Target_encoded,1,RFC)

            st.metric("Accuracy",str(round(acc,4)*100)+"%")
            plt.figure(figsize=(3,2))
            fig6 = sns.heatmap(cfm,annot=True,cmap="rocket_r")
            st.pyplot(plt.gcf())
        st.write("### The Random Forest Algorithm 🌳🌳🌳🌴🌲🎄..... performs better than the remaining algorithms")
    
    with tab3:
        folds = st.number_input("Please enter the number of folds/splits",value=5,min_value=5,max_value=20,step=1)
        kf = KFold(n_splits=folds,shuffle=True,random_state=42)
        st.header("K-Nearest Neighbors Classifier")
        temp=[]
        knn = KNeighborsClassifier(n_neighbors=5)
        ss = StandardScaler()
        for train_index, test_index in kf.split(X):
            X_train,X_test = X.iloc[train_index,:],X.iloc[test_index,:]
            y_train,y_test = Target_encoded.iloc[train_index,],Target_encoded.iloc[test_index,]
            ss.fit(X_train)
            X_train_scaled = ss.transform(X_train)
            X_test_scaled = ss.transform(X_test)
            acc = helper_2(X_train_scaled, X_test_scaled,y_train,y_test,knn)
            temp.append(acc)
        temp_array = np.array(temp)
        st.write("The Obtained Accuracy in each fold")
        st.write(pd.DataFrame(temp_array.reshape(folds,1),columns=["Accuracy"]))
        st.metric("The average accuracy is: ",str(round(temp_array.mean()*100,2))+"%")
        st.header("Decision Tree Classifier 🌳")
        temp=[]
        DTC = DecisionTreeClassifier(random_state=42)
        for train_index, test_index in kf.split(X):
            X_train,X_test = X.iloc[train_index,:],X.iloc[test_index,:]
            y_train,y_test = Target_encoded.iloc[train_index,],Target_encoded.iloc[test_index,]
            acc = helper_2(X_train, X_test,y_train,y_test,DTC)
            temp.append(acc)
        temp_array = np.array(temp)
        st.write("The Obtained Accuracy in each fold")
        st.write(pd.DataFrame(temp_array.reshape(folds,1),columns=["Accuracy"]))
        st.metric("The average accuracy is: ",str(round(temp_array.mean()*100,2))+"%")
        st.header("Random Forest Classifier 🌳🌳🌳🌳........")
        temp=[]
        RFC = RandomForestClassifier(random_state=42)
        for train_index, test_index in kf.split(X):
            X_train,X_test = X.iloc[train_index,:],X.iloc[test_index,:]
            y_train,y_test = Target_encoded.iloc[train_index,],Target_encoded.iloc[test_index,]
            acc = helper_2(X_train, X_test,y_train,y_test,RFC)
            temp.append(acc)
        temp_array = np.array(temp)
        st.write("The Obtained Accuracy in each fold")
        st.write(pd.DataFrame(temp_array.reshape(folds,1),columns=["Accuracy"]))
        st.metric("The average accuracy is: ",str(round(temp_array.mean()*100,2))+"%")
        st.write("### Again the Random Forest Algorithm is the best estimator among three estimators")
    with tab4:
        st.write("#### Here, I'm considering some important parameters of each algorithm for Hyperparameter Tuning")
        st.write("Excited to see the results with best parameters & best scores😱😱😱🤩🤩")
        imp_list_of_options=["Static results (The results of Hyperparamter Tuning stored by setting a fixed random seed)"
                                             ,"Dynamic Results (The entire Hyper paramter tuning code runs, Time Consuming process)"]
        optionht = st.radio("Please select one option",["Static results (The results of Hyperparamter Tuning stored by setting a constant random seed)"
                                             ,"Dynamic Results (The entire Hyper paramter tuning code runs, Time Consuming process)"],index=0)
        if optionht==imp_list_of_options[0]:
           st.write(pd.DataFrame({"Classification Models/Estimators"
                                  :["KNearestNeighbors","RandomForest","DecisionTree"],
                                  "Best Scores (in %)":[0.9384*100,0.9796*100,0.9795*100]}))
           st.write("### The following are the best paramters of K Nearest Neighbors Classifier algorithm")
           st.text({'metric': 'manhattan', 'n_neighbors': 5})
           st.write("### The following are the best paramters of Random Forest Classifier algorithm")
           st.text({'criterion': 'gini', 'max_depth': 1, 'n_estimators': 50})
           st.write("### The following are the best paramters of Decision Tree Classifier algorithm")
           st.text({'criterion': 'gini', 'max_depth': 3})
           st.write("## So, finally our best model is RandomForest Classifier Algorithm, which acheives a best score of 97.96%")
           st.write("### Finally, I will use the RandomForest Algorithm for the Real-time data predictions.")
        elif optionht==imp_list_of_options[1]:
            model_params = {
                        'KNN': {
                        'model': KNeighborsClassifier(),
                        'params' : {
                            'n_neighbors': [3,5,7],
                            "metric":["cosine","manhattan","euclidean"]
                        }  
                        },
                        'random_forest': {
                        'model': RandomForestClassifier(random_state=42),
                        'params' : {
                            'n_estimators': [50,100,150],
                            "criterion" : ["gini", "entropy", "log_loss"],
                            "max_depth":list(range(1,4))
                        }
                        },
                        'Decision_Tree' : {
                        'model':DecisionTreeClassifier(random_state=42),
                        'params': {
                            "criterion" : ["gini", "entropy", "log_loss"],
                            "max_depth":list(range(1,4))
                        }
                        }
                        }
            scores = []
            parameters=[]
            knn,rfc,dtc = model_params.items()
            
            gscv1 =  GridSearchCV(knn[1]["model"],knn[1]["params"], cv=5, return_train_score=False)
            ss = StandardScaler()
            X_scaled = ss.fit_transform(X)
            gscv1.fit(X_scaled, Target_encoded)
            scores.append({
            'model': knn[0],
            'best_score': gscv1.best_score_,
            })
            parameters.append(gscv1.best_params_)
            
            gscv2 =  GridSearchCV(rfc[1]["model"],rfc[1]["params"], cv=5, return_train_score=False)
            gscv2.fit(X, Target_encoded)
            scores.append({
            'model': rfc[0],
            'best_score': gscv2.best_score_,
         
            })
            parameters.append(gscv2.best_params_)
            
            
            gscv3 =  GridSearchCV(dtc[1]["model"],dtc[1]["params"], cv=5, return_train_score=False)
            gscv3.fit(X, Target_encoded)
            scores.append({
            'model': dtc[0],
            'best_score': gscv3.best_score_,
          
            })
            parameters.append(gscv3.best_params_)
            
            my_hy_data  = pd.DataFrame(scores,columns=['model','best_score'])
            st.write(my_hy_data)
            st.write("### The following are the best paramters of K Nearest Neighbors Classifier algorithm")
            st.text(parameters[0])
            st.write("### The following are the best paramters of Random Forest Classifier algorithm")
            st.text(parameters[1])
            st.write("### The following are the best paramters of Decision Tree Classifier algorithm")
            st.text(parameters[2])
            st.write("## So, finally our best model is RandomForest Classifier Algorithm, which acheives a best score of {}%".format(round(gscv2.best_score_*100,2)))
            st.write("### Finally, I will use the RandomForest Algorithm for the Real-time data predictions.")
        
elif optionm=="Forest Fires Prediction":
    col1,col2,col3,col4,col5 = st.columns(5,gap="large")
    with col1:
        v1 = st.slider(label = X.columns[0],value = float(X[X.columns[0]].min()),min_value = X[X.columns[0]].min(),max_value=X[X.columns[0]].max(),step=0.1)
    with col2:
        v2 = st.slider(label = X.columns[1],value = float(X[X.columns[1]].min()),min_value = X[X.columns[1]].min(),max_value=X[X.columns[1]].max(),step=0.1)
    with col3:
        v3 = st.slider(label = X.columns[2],value = float(X[X.columns[2]].min()),min_value = X[X.columns[2]].min(),max_value=X[X.columns[2]].max(),step=0.1)
    with col4:
        v4 = st.slider(label = X.columns[3],value = float(X[X.columns[3]].min()),min_value = X[X.columns[3]].min(),max_value=X[X.columns[3]].max(),step=0.1)
    with col5:
        v5 = st.slider(label = X.columns[4],value = float(X[X.columns[4]].min()),min_value = X[X.columns[4]].min(),max_value=X[X.columns[4]].max(),step=0.1)
    col6,col7,col8,col9,col10 = st.columns(5,gap="large")
    with col6:
        v6 = st.slider(label = X.columns[5],value = float(X[X.columns[5]].min()),min_value = X[X.columns[5]].min(),max_value=X[X.columns[5]].max(),step=0.1)
    with col7:
        v7 = st.slider(label = X.columns[6],value = float(X[X.columns[6]].min()),min_value = X[X.columns[6]].min(),max_value=X[X.columns[6]].max(),step=0.1)
    with col8:
        v8 = st.slider(label = X.columns[7],value = float(X[X.columns[7]].min()),min_value = X[X.columns[7]].min(),max_value=X[X.columns[7]].max(),step=0.1)
    with col9:
        v9 = st.slider(label = X.columns[8],value = float(X[X.columns[8]].min()),min_value = X[X.columns[8]].min(),max_value=X[X.columns[8]].max(),step=0.1)
    with col10:
        v10 = st.slider(label = X.columns[9],value = float(X[X.columns[9]].min()),min_value = X[X.columns[9]].min(),max_value=X[X.columns[9]].max(),step=0.1)
        
    if st.button('Predict'):
        rf = RandomForestClassifier(criterion= 'gini', max_depth= 3, n_estimators= 50,random_state=42)
        x_train,x_test,y_train,y_test = train_test_split(X,Target_encoded,test_size=0.3,random_state=42)
        rf.fit(x_train,y_train)
        df1 = pd.DataFrame([v1,v2,v3,v4,v5,v6,v7,v8,v9,v10]).transpose()
        df2 = df1.iloc[0,:].astype(float)
        prediction = rf.predict(df1)
        if int(prediction[0])==0:
           st.write("There is no fire occurrence")
           st.image("nofire.png",width=200)
        elif int(prediction[0])==1:
           st.write("There is an occurrence of fire")
           st.image("fire.jpg",width=200)

elif optionm=="The Power of PCA":
    st.write("## 👉 One of the main purposes of the PCA is the dimensionality reduction😁😁")
    st.write("#### 👉 Please tell, How much percentage of information you want to to capture from the dataset, Don't be greedy, If you select 100% 😥, there is no point of dimensionality reduction 🤔🤔. You can go with the same dataset what you have!!😁😁😁")
    per = st.number_input("Input"
                            ,value=0.95,max_value=0.99,min_value=0.10)
    st.write("##### 👉 Based on that, I would like to generate a new dataset with reduced number of features🤩🤩")
    st.write("##### 👉 Again here, I will fit to our best model which we find from hyperparameter tuning with the new dataset and I will provide the accuracy scores.")
    st.write("##### 👉 Interesting right 😉!! Let's see the results 😎")
    ss = StandardScaler()
    X_scaled = ss.fit_transform(X)
    pca = PCA(per)
    pca_data = pca.fit_transform(X_scaled)
    x_train_pca,x_test_pca,y_train_pca,y_test_pca = train_test_split(pca_data,Target_encoded,test_size=0.3,random_state=42)
    rfc_pca = RandomForestClassifier(criterion= 'gini', max_depth= 3, n_estimators= 50,random_state=42)
    rfc_pca.fit(x_train_pca,y_train_pca)
    st.write("##### 👉 The number of features in the new dataset after dimensionality reduction are:")
    st.metric("New Features",pca.n_components_)
    predictions = rfc_pca.predict(x_test_pca)
    st.write("##### 👉 The accuracy achieved by the model with the new features:")
    st.metric("Accuracy",str(round(accuracy_score(y_test_pca,predictions)*100,4))+"%")
    st.write("##### 👉 Both Accuracy and the dimesnionality reduction are trade-off. If we do the dimensionality reduction then we should compromise the accuracy.")
    
    
elif optionm=="👉 Feel free for mining to discover more insights 😀🤔😁😱😎":
    st.markdown(" ## **Please select any two features on which you want to do the analysis** ")
    optiona = st.radio("Select the type of analysis, you want to perform", ["Quantitative Vs Quantitative features","Quantitative Vs Qualitative features"])
    
    
    if optiona=="Quantitative Vs Quantitative features":
        col1,col2,col3 = st.columns(3,gap="large")
        with col1:
            optionx = st.radio(
                   'Select one feature which you want to see on x?',
                   numerical_columns
                   ,index=0)
            st.write('Your selected feature:', optionx)
        with col2:
            optiony = st.radio(
                 'Select one feature which you want to see on y?',
                 numerical_columns,index=1)
            st.write('Your selected feature:', optiony)
        with col3:
            optionp = st.selectbox(
         'please select the plot to analyze the data?',
         ["scatter","distribution","histogram","combinational plots"],index=3)
            st.write('Your selected plot:', optionp)
        optionc1 = categorical_columns[0]
        if optionp=="scatter":
            st.write('''
                  # Scatter plot
                  ''')
            sns.scatterplot(data=forest_fires,x=optionx,y=optiony,hue=optionc1)
            st.pyplot(plt.gcf())
        elif optionp=="distribution":
            st.write('''
                  # Distribution plot
                  ''')
            sns.displot(data=forest_fires.loc[:,[optionx,optiony]],kind="kde")
            st.pyplot(plt.gcf())
        elif optionp=="histogram":
            st.write('''
                  # Histogram plot
                  ''')
         
            sns.histplot(data=forest_fires.loc[:,[optionx,optiony]])
            st.pyplot(plt.gcf())
        elif optionp=="combinational plots":
            st.write('''
                   # Joint plot
                   ''')
            sns.jointplot(data=forest_fires,x=optionx,y=optiony,hue=optionc1)
            st.pyplot(plt.gcf())
    
    
    if optiona=="Quantitative Vs Qualitative features":
        col4, col5 = st.columns(2,gap="large")
        with col4:
            pt=st.selectbox(
     'please select the plot to analyze the data?',
     ["strip","boxplot","violinplot"],index=0)
            st.write('Your selected plot:', pt)
        with col5:
            optionc2 = st.selectbox(
     'please choose one feature for categorical plot',
     numerical_columns,index=4)
            st.write('Your selected feature:', optionc2)
        optionc1 = categorical_columns[0]
        if pt=="strip":
            
            st.write('''
                  # Strip plot
                  ''')
            
            cp1 = px.strip(forest_fires, y=optionc1, x=optionc2, orientation="h",color=optionc1,width=950, height=500)
            st.plotly_chart(cp1)
        elif pt=="boxplot":
            st.write('''
                  # Box plot
                  ''')
            cp2 = px.box(forest_fires, y=optionc1, x=optionc2, color=optionc1, width=950, height=500)
            st.plotly_chart(cp2)
        elif pt=="violinplot":
            sns.violinplot(data=forest_fires,x=optionc2,y=optionc1)
            st.pyplot(plt.gcf())
    
    
elif optionm=="Exploratory Data Analysis":
    st.write("##### 👉 If you want to see my work and want to contribute towards improvements, Feel free to use the below colab and github links")
    st.write("[Project_Forest_Fires_Data_Cleaning](https://colab.research.google.com/drive/1alWrrxH_Hp_1s--hUYEon5TOpNCJ8UI1?usp=sharing)")
    st.write("[Forest_Fires_Exploratory_Data_Analysis](https://colab.research.google.com/drive/1e0vXykPnHDJDMOL69bZA29c5_PD28cKu?usp=sharing)")
    st.write("[GitHub](https://github.com/Nitish-Satya-Sai/Project_Forest_Fires.git)")
    col6,col7 = st.columns(2,gap="medium")
    with col6:
        imj1 = Image.open("1.png")
        st.image(imj1)
    with col7:
        imj2 = Image.open("2.png")
        st.image(imj2)
    col8,col9 = st.columns(2,gap="medium")
    with col8:
        imj1 = Image.open("3.png")
        st.image(imj1)
    with col9:
        imj2 = Image.open("4.png")
        st.image(imj2)
    col10,col11 = st.columns(2,gap="medium")
    with col10:
        imj1 = Image.open("5.png")
        st.image(imj1)
    with col11:
        imj2 = Image.open("6.png")
        st.image(imj2)
    col12,col13 = st.columns(2,gap="medium")
    with col12:
        imj1 = Image.open("7.png")
        st.image(imj1)
    with col13:
        imj2 = Image.open("8.png")
        st.image(imj2)
    col14,col15 = st.columns(2,gap="medium")
    with col14:
        imj1 = Image.open("9.png")
        st.image(imj1)
    with col15:
        imj2 = Image.open("10.png")
        st.image(imj2)
    col16,col17 = st.columns(2,gap="medium")
    with col16:
        imj1 = Image.open("11.png")
        st.image(imj1)
    with col17:
        imj2 = Image.open("12.png")
        st.image(imj2)
    col18,col19 = st.columns(2,gap="medium")
    with col18:
        imj1 = Image.open("13.png")
        st.image(imj1)
    with col19:
        imj2 = Image.open("14.png")
        st.image(imj2)
    col20,col21 = st.columns(2,gap="medium")
    with col20:
        imj1 = Image.open("15.png")
        st.image(imj1)
    with col21:
        imj2 = Image.open("16.png")
        st.image(imj2)
    col22,col23 = st.columns(2,gap="medium")
    with col22:
        imj1 = Image.open("17.png")
        st.image(imj1)
    with col23:
        imj2 = Image.open("18.png")
        st.image(imj2)
    col24,col25 = st.columns(2,gap="medium")
    with col24:
        imj1 = Image.open("19.png")
        st.image(imj1)
    with col25:
        imj2 = Image.open("20.png")
        st.image(imj2)
    col26,col27 = st.columns(2,gap="medium")
    with col26:
        imj1 = Image.open("21.png")
        st.image(imj1)
    with col27:
        imj2 = Image.open("22.png")
        st.image(imj2)
