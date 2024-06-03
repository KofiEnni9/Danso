import pandas as pd
import json

xls = pd.ExcelFile('2009 NSMQ contest 1.xlsx')

for sheet_name in xls.sheet_names:
    df = pd.read_excel(xls, sheet_name=sheet_name)
    data_dict = df.to_dict(orient='records')
    

    json_filename = f'{sheet_name}.json'
    with open(json_filename, 'w') as f:
        json.dump(data_dict, f, indent=4)
        
    print(f'Converted {sheet_name} sheet to {json_filename}')
