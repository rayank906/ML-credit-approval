import pandas as pd
import numpy as np

#load data
app = pd.read_csv("application_record.csv")
credit = pd.read_csv("credit_record.csv")


#fade duplicates fr
app = app.drop_duplicates(subset=['ID'], keep='first', ignore_index=True)

#change M/F --> 0/1
if app['CODE_GENDER'].dtype == object:
    app['CODE_GENDER'] = app['CODE_GENDER'].map({'M': 0, 'F': 1})
else:
    app['CODE_GENDER'] = app['CODE_GENDER'].astype(int)

#change car from y/n to 0/1
app['FLAG_OWN_CAR'] = app['FLAG_OWN_CAR'].map({'N': 0, 'Y': 1})

#change realty from y/n to 0/1
app['FLAG_OWN_REALTY'] = app['FLAG_OWN_REALTY'].map({'N': 0, 'Y': 1})

#map family status
app['NAME_FAMILY_STATUS'] = app['NAME_FAMILY_STATUS'].map({
    'Single / not married': 0,
    'Married': 1,
    'Separated': 2,
    'Civil marriage': 3,
    'Widow': 4
})

# encode income type
app['NAME_INCOME_TYPE'] = app['NAME_INCOME_TYPE'].map({
    'Working': 0,
    'Commercial associate': 1,
    'Pensioner': 2,
    'State servant': 3,
    'Student': 4
})

# encode education
app['NAME_EDUCATION_TYPE'] = app['NAME_EDUCATION_TYPE'].map({
    'Lower secondary': 0,
    'Secondary / secondary special': 1,
    'Incomplete higher': 2,
    'Higher education': 3,
    'Academic degree': 4
})

# name housing type
app['NAME_HOUSING_TYPE'] = app['NAME_HOUSING_TYPE'].map({
    'With parents': 0,
    'Rented apartment': 1,
    'Municipal apartment': 2,
    'Co-op apartment': 3,
    'House / apartment': 4,
    'Office apartment': 5
})

# occupation type
app['OCCUPATION_TYPE'] = app['OCCUPATION_TYPE'].fillna('Unknown')
occupation_map = {
    'Laborers': 0,
    'Core staff': 1,
    'Sales staff': 2,
    'Managers': 3,
    'Drivers': 4,
    'High skill tech staff': 5,
    'Accountants': 6,
    'Medicine staff': 7,
    'Cooking staff': 8,
    'Security staff': 9,
    'Cleaning staff': 10,
    'Private service staff': 11,
    'Low-skill Laborers': 12,
    'Waiters/barmen staff': 13,
    'Secretaries': 14,
    'Realty agents': 15,
    'HR staff': 16,
    'IT staff': 17,
    'Unknown': 18
}
app['OCCUPATION_TYPE'] = app['OCCUPATION_TYPE'].map(occupation_map)

# these bums did not work 1000 yrs, must be unemployed --> 0 yrs
app['DAYS_EMPLOYED'] = app['DAYS_EMPLOYED'].replace(365243, 0)

# convert birthday to yrs
app['AGE_YEARS'] = (-app['DAYS_BIRTH'] / 365.25).round(1)
app = app.drop(columns=['DAYS_BIRTH'])

# days employed to years employed
app['YEARS_EMPLOYED'] = (-app['DAYS_EMPLOYED'] / 365.25).round(1)
app = app.drop(columns=['DAYS_EMPLOYED'])

# flag_mobil 1 for everyting , no output on our code
app = app.drop(columns=['FLAG_MOBIL'])

# cast to int to prevent hella ugly table 
app['CNT_FAM_MEMBERS'] = app['CNT_FAM_MEMBERS'].astype(int)

# ============================================================
# 3. BUILD TARGET VARIABLE FROM CREDIT RECORD
# ============================================================
# STATUS codes:
#   C = paid off that month
#   X = no loan for that month
#   0 = 1-29 days past due
#   1 = 30-59 days past due
#   2 = 60-89 days past due
#   3 = 90-119 days past due
#   4 = 120-149 days past due
#   5 = 150+ days past due (written off as bad debt)
#
# Target: 1 = risky (ever had status 2, 3, 4, or 5 — i.e. 60+ days overdue)
#         0 = safe

credit['STATUS'] = credit['STATUS'].replace({'C': 0, 'X': 0})
credit['STATUS'] = credit['STATUS'].astype(int)

# Flag as risky if any month had 60+ days overdue (status >= 2)
credit_agg = credit.groupby('ID').agg(
    months_on_record=('MONTHS_BALANCE', 'count'),
    max_dpd=('STATUS', 'max'),
    avg_dpd=('STATUS', 'mean'),
    num_late_payments=('STATUS', lambda x: (x >= 1).sum())
).reset_index()

credit_agg['TARGET'] = (credit_agg['max_dpd'] >= 2).astype(int)

#merge the 2 files
df = app.merge(credit_agg, on='ID', how='inner')

#save outputs

# Save the cleaned application-only file (all applicants)
app.to_csv("cleaned_application_record.csv", index=False)

# Save the final ML-ready merged dataset
df.to_csv("ml_ready_dataset.csv", index=False)

print(f"Cleaned application records: {len(app)} rows, {len(app.columns)} cols")
print(f"ML-ready dataset (merged):   {len(df)} rows, {len(df.columns)} cols")
print(f"Target distribution:\n{df['TARGET'].value_counts()}")
print(f"\nColumns: {list(df.columns)}")
