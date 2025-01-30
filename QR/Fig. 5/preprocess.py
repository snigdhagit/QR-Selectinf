import numpy as np
import pandas as pd

colspecs = [(8, 12), (12, 14), (18, 22), (22, 23), (49, 50),
            (74, 76), (83, 84), (103, 104),         (118, 119),
            (119, 120), (123, 124), (146, 148),          (162, 163),
            (170, 172), (172, 174), (174, 176), (178, 179), (181, 182),
            (197, 200), (205, 208), (223, 225), (237, 239), (251, 252),
            (252, 254), (279, 281), (282, 286), (291, 294), (298, 301),
            (254, 256), (256, 258), (258, 260),
            (303, 305), (312, 313), (313, 314), (314, 315), (315, 316),
            (316, 317), (317, 318), (324, 325), (325, 326), (326, 327),
            (331, 333), (352, 353),
                        (359, 360), (360, 361), (382, 383), (383, 384),
            (384, 385), (385, 386), (386, 387), (387, 388), (400, 401),
            (401, 402), (407, 408), (426, 427),
                                    (432, 433), (433, 434), (435, 436),
            (443, 445), (447, 449), (453, 454), (455, 456), (474, 475),
            (475, 476), (489, 491), (497, 500), (503, 507), (516, 517),
            (517, 518), (518, 519), (519, 520), (520, 521), (521, 522),
            (560, 561),
                                     (566, 567), (567, 568), (568, 569),
            (116, 117), (161, 162)]

colnames = ['Birth Year', 'Birth Month', 'Time of Birth', 'Birth Day of Week', 'Birth Place',
            'Mother’s Age', 'Mother’s Nativity', 'Residence Status',               'Paternity Acknowledged',
            'Marital Status', 'Mother’s Education', 'Father’s Age',                'Father’s Education',
            'Prior Births Now Living', 'Prior Births Now Dead', 'Prior Other Terminations', 'Live Birth Order', 'Total Birth Order',
            'Interval Since Last Live Birth', 'Interval Since Last Other Pregnancy', 'Month Prenatal Care Began', 'Number of Prenatal Visits', 'WIC',
            'Cigarettes Before Pregnancy', 'Mother’s Height', 'Body Mass Index', 'Pre-pregnancy Weight', 'Delivery Weight',
            'Cigarettes 1st Trimester', 'Cigarettes 2nd Trimester', 'Cigarettes 3rd Trimester',
            'Weight Gain', 'Pre-pregnancy Diabetes', 'Gestational Diabetes', 'Pre-pregnancy Hypertension', 'Gestational Hypertension',
            'Hypertension Eclampsia', 'Previous Preterm Birth', 'Infertility Treatment Used', 'Fertility Enhancing Drugs', 'Asst. Reproductive Technology',
            'Number of Previous Cesareans', 'No Infections Reported',
                            'Successful External Cephalic Version', 'Failed External Cephalic Version', 'Induction of Labor', 'Augmentation of Labor',
            'Steroids', 'Antibiotics', 'Chorioamnionitis', 'Anesthesia', 'Fetal Presentation at Delivery',
            'Final Route & Method of Delivery', 'Delivery Method', 'No Maternal Morbidity Reported',
                            'Attendant at Birth', 'Mother Transferred', 'Payment',
            'Five Minute APGAR Score', 'Ten Minute APGAR Score', 'Plurality', 'Plurality Imputed', 'Sex of Infant',
            'Imputed Sex', 'Combined Gestation', 'Obstetric Estimate Edited', 'Birth Weight', 'Assisted Ventilation (immediately)',
            'Assisted Ventilation > 6 hrs', 'Admission to NICU', 'Surfactant', 'Antibiotics for Newborn', 'Seizures',
            'No Congenital Anomalies Checked',
                            'Infant Transferred', 'Infant Living at Time of Report', 'Infant Breastfed at Discharge',
            'Mother’s Race/Hispanic Origin', 'Father’s Race/Hispanic Origin']

# data = pd.read_fwf('Nat2022PublicUS.c20230504.r20230822.txt',
#                    header=None,
#                    nrows=10000,
#                    colspecs=colspecs,
#                    names=colnames)

chunk_size = 2000
data = pd.DataFrame()
for chunk in pd.read_fwf('Nat2022PublicUS.c20230504.r20230822.txt', header=None, colspecs=colspecs, names=colnames, chunksize=chunk_size):
    filtered_chunk = chunk[chunk['Plurality'] == 2]
    data = pd.concat([data, filtered_chunk])
# data.to_csv('./results_birth/selected_data.csv', index=False)
#
# data = pd.read_csv('./results_birth/selected_data.csv', low_memory=False)
# print(data.shape)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Remove the rows with 'U' / Unknown
cols_categorical = data.select_dtypes(include=['object']).columns
rows_unknown = data[cols_categorical].apply(lambda x: x.str.contains('U', 'X')).any(axis=1)
data = data[~rows_unknown]
data = data[data['Mother’s Nativity'] != 3]
data = data[data['Residence Status'] == 1]
data = data[data['Mother’s Education'] != 9]
data = data[data['Father’s Age'] != 99]
data = data[data['Father’s Education'] != 9]
data = data[data['Prior Births Now Living'] != 99]
data = data[data['Prior Births Now Dead'] != 99]
data = data[data['Prior Other Terminations'] != 99]
data = data[data['Interval Since Last Live Birth'] != 999]
data = data[data['Interval Since Last Other Pregnancy'] != 999]
data = data[data['Month Prenatal Care Began'] != 99]
data = data[data['Number of Prenatal Visits'] != 99]
data = data[data['Cigarettes Before Pregnancy'] != 99]
data = data[data['Mother’s Height'] != 99]
data = data[data['Pre-pregnancy Weight'] != 999]
data = data[data['Delivery Weight'] != 999]
data = data[data['Weight Gain'] != 999]
data = data[data['Number of Previous Cesareans'] != 99]
data = data[data['Fetal Presentation at Delivery'] != 9]
data = data[data['Final Route & Method of Delivery'] != 9]
data = data[data['Payment'] != 9]
data = data[data['Five Minute APGAR Score'] != 99]
data = data[data['Ten Minute APGAR Score'] != 99]
data = data[data['No Infections Reported'] != 9]
data = data[data['No Maternal Morbidity Reported'] != 9]
data = data[data['No Congenital Anomalies Checked'] != 9]
data = data[data['Mother’s Race/Hispanic Origin'] != 8]
data = data[data['Father’s Race/Hispanic Origin'] != 9]

# Remove the columns with 'X'
data = data.drop(columns=['Fertility Enhancing Drugs', 'Asst. Reproductive Technology'])

# Add season variable
def get_season(month):
    if month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Autumn'
    elif month in [12, 1, 2]:
        return 'Winter'
data['Birth Season'] = data['Birth Month'].apply(get_season)
data = data.drop(columns=['Birth Month'])

# One-hot encoding for categoical variable
cols_one_hot = ['Mother’s Education', 'Father’s Education',
                'Mother’s Race/Hispanic Origin', 'Father’s Race/Hispanic Origin',
                'Fetal Presentation at Delivery', 'Delivery Method', 'Payment', 'Birth Season']
for col in cols_one_hot:
    dummies = pd.get_dummies(data[col], prefix=col, drop_first=True)
    data = pd.concat([data, dummies], axis=1)
    data = data.drop(columns=col)

# One-hot encoding for binary variable
cols_binary = data.select_dtypes(include=['object']).columns
for col in cols_binary:
    # print(col)
    # print(data[col].astype('category').cat.categories[0])
    data[col] = data[col].astype('category').cat.codes


# Remove the columns with one unique value
cols_unique = [col for col in data.columns if data[col].nunique() == 1]
data = data.drop(columns=cols_unique)
data = data.dropna(axis=1)

# Calculating the correlation matrix
data = data.drop(columns=['Body Mass Index', 'Final Route & Method of Delivery',
                          'Interval Since Last Live Birth', 'Interval Since Last Other Pregnancy'])
#? Interval Since Last Live Birth
#? Interval Since Last Other Pregnancy
corr = data.select_dtypes(include=['int64', 'float64']).corr()
pairs = corr.stack().reset_index()
pairs.columns = ['Feature 1', 'Feature 2', 'Correlation']
pairs = pairs[(pairs['Correlation'] > 0.7) & (pairs['Feature 1'] < pairs['Feature 2'])]
print(pairs)
data = data.drop(columns=['Obstetric Estimate Edited', 'Live Birth Order', 'Delivery Weight'])

# Remove the columns where one of the value counts is smaller than 150
cols_binary = data.columns[(data.nunique() == 2) & (data.isin([0, 1]).all())]
cols_small = [col for col in cols_binary if data[col].value_counts().min() <= 150]
print(cols_small)
data = data.drop(columns=cols_small)
data.to_csv('cleaned_data.csv', index=False)