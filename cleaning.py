import pandas as pd 

application_record_df = pd.read_csv("application_record.csv")

#deleting duplicates
cleaned_df = application_record_df.drop_duplicates(subset=['ID'], keep='first', ignore_index=True)

#changing M/F --> 0/1
cleaned_df['CODE_GENDER'] = cleaned_df['CODE_GENDER'].map({'M':0, 'F':1})

#change marital status
cleaned_df['NAME_FAMILY_STATUS'] = cleaned_df['NAME_FAMILY_STATUS'].map({'Single / not married':0, 'Married':1, 'Separated':2, 'Civil Marriage':3})



#updating removed csv file 
cleaned_df.to_csv("cleaned_application_record.csv", index=False)







    


