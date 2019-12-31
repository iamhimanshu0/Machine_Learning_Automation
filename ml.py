import streamlit as st
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score


def main():
	df = ''
	st.title("Machine/Data Science Learning Automation")
	st.sidebar.title("Himanshu Tripathi")
	st.sidebar.header("Machine Learning Automation")
	activites = ['EDA','PLOT','MODEL BUILDING','ABOUT']
	choice = st.sidebar.selectbox("Select Actvity", activites)

	

	data = st.file_uploader("Upload Dataset", type=['csv','txt',])
	if data is not None:
		df = pd.read_csv(data)
		st.success("Data File Uploaded Successfully")

	# Data Model Building
	X = ""
	y = ""
	X_train=''
	X_test=''
	y_train=''
	y_test = ''
	y_pred = ''		
		

	if choice == 'EDA':
			st.subheader("Exploratory Data Analysis")
			# Data Show
			if st.checkbox("Show Data"):
				select_ = st.radio("HEAD OR TAIL",('All','HEAD','TAIL'))
				if select_ == 'All':
					st.dataframe(df)
				elif select_ == 'HEAD':
					st.dataframe(df.head())
				elif select_ == 'TAIL':
					st.dataframe(df.tail())
			# Columns
			if st.checkbox("Show Columns"):
				select_ = st.radio("Select Columns",('All Columns','Specific Column'))
				if select_ == "All Columns":
					st.write(df.columns)
				if select_ == "Specific Column":
					col_spe = st.multiselect("Select Columns To Show",df.columns)
					st.write(df[col_spe])

			# Show Dimension
			if st.checkbox("Show Dimension"):
				select_ = st.radio('Select Dimension',('All','Row','Column'))
				if select_ == "All":
					st.write(df.shape)
				elif select_ == "Row":
					st.write(df.shape[0])
				elif select_ == "Column":
					st.write(df.shape[1])

			# Summary of dataset
			if st.checkbox("Summary of Data Set"):
				st.write(df.describe())


			# Value Counts
			if st.checkbox("Value Count"):
				select_ = st.multiselect("Select values",df.columns.tolist())
				st.write(df[select_].count())


			# Show data Type
			if st.checkbox("Show Data Type"):
				select_ = st.radio("Select ",('All Columns','Specific Column'))
				if select_ == "All Columns":
					st.write(df.dtypes)
				elif select_ == "Specific Column":
					s = st.multiselect("Select value",df.columns.tolist())
					st.write(df[s].dtypes)

			# Check for Null Values
			if st.checkbox("Check For Null Values"):
				st.write(df.isnull().sum())

	#Data Visualization 

	elif choice == 'PLOT':
		st.subheader("Data Visualization")
		if st.checkbox("Show Data"):
			select_ = st.radio("HEAD OR TAIL",('All','HEAD','TAIL'))
			if select_ == 'All':
				st.dataframe(df)
			elif select_ == 'HEAD':
				st.dataframe(df.head())
			elif select_ == 'TAIL':
				st.dataframe(df.tail())
		# Show Columns Names
		if st.checkbox("Columns Names"):
			st.write(df.columns)
		# Check for null values in the form of HeatMap
		if st.checkbox("Show Null Values in Heatmap"):
			st.write(sns.heatmap(df.isnull()))
			st.pyplot()

		# Quick Analysis
		if st.checkbox("Quick Analysis"):
			select_ = st.radio("Select Type for Quick Analysis",('Count Plot','Box Plot','Bar Plot for Specific Columns','lmplot','Scatter Plot','Correlation Heatmap','Histogram','Joint Distribution Plot'))
			if select_ == "Count Plot":
				st.write(df.dtypes)
				s = st.text_input('Enter Column Name')
				try:
					if s != " ":
						st.write(sns.countplot(df[s]))
						st.pyplot()
				except Exception as e:
					st.error(e)
			elif select_ == 'lmplot':
				st.write(df.dtypes)
				x = st.text_input('Enter X Value')
				y = st.text_input("Enter Y Value")
				try:
					if x != " " and y != " ":
						st.write(x,y)
						st.write(sns.lmplot(x,y,data=df))
						st.pyplot()
				except Exception as e:
					st.error(e)

			elif select_ == 'Scatter Plot':
				st.write(df.dtypes)
				x = st.text_input('Enter X Value')
				y = st.text_input("Enter Y Value")
				try:
					if x != " " and y != " ":
						st.write(x,y)
						st.write(sns.scatterplot(x,y,data=df))
						st.pyplot()
				except Exception as e:
					st.error(e)
				
			elif select_ == 'Box Plot':
				st.write(sns.boxplot(data=df))
				st.pyplot()

			elif select_ == "Bar Plot for Specific Columns":
				x = st.multiselect('Select Value',df.columns)
				try:
					if x != " ":
						st.write(sns.barplot(data=df[x]))
						st.pyplot()
				except Exception as e:
					st.error(e)

			elif select_ == "Correlation Heatmap":
				st.write(sns.heatmap(df.corr()))
				st.pyplot()

			elif select_ == "Histogram":
				x = st.multiselect('Select Numerical Variables',df.columns)
				try:
					if x != " ":
						st.write(sns.distplot(df[x]))
						st.pyplot()
				except Exception as e:
					st.error(e)

			elif select_ == "Joint Distribution Plot":
				st.write(df.dtypes)
				x = st.text_input('Enter X Value')
				y = st.text_input("Enter Y Value")
				try:
					if x != " " and y != " ":
						st.write(x,y)
						st.write(sns.jointplot(x,y,data=df))
						st.pyplot()
				except Exception as e:
					st.error(e)

				# st.write(sns.countplot(df[str(s)]))
				# st.pyplot()



	# Model Building
	elif choice == 'MODEL BUILDING':
		st.subheader("MODEL BUILDING")

		if st.checkbox("Show Null Values"):
			st.write(df.isnull().sum())

		if st.checkbox("Fill null Values"):
			select_ = st.radio("Choose",('Mean Value','ffill Method','sel'))
			
			sel_column = st.multiselect("Select Column",df.columns)
			if select_ == 'Mean Value':
				df[sel_column] = df[sel_column].fillna(df[sel_column].mean(),inplace=True)
				st.write(df.isnull().sum())
			elif select_ == 'ffill Method':
				df[sel_column] = df[sel_column].fillna(method='ffill',inplace=True) 














	elif choice == 'ABOUT':
		st.subheader("About Me")










if __name__ == "__main__":
	main()