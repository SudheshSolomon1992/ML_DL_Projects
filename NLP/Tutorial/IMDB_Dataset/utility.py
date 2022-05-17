import pandas as pd
from tabulate import tabulate

result_dict = {}

if 'Algorithm' not in result_dict.keys():
    result_dict['Algorithm'] = list()

if 'Time (s)' not in result_dict.keys():
    result_dict['Time (s)'] = list()

if 'Train Score' not in result_dict.keys():
    result_dict['Train Score'] = list()

if 'Test Score' not in result_dict.keys():
    result_dict['Test Score'] = list()

if 'Overfitting' not in result_dict.keys():
    result_dict['Overfitting'] = list()

def print_scores(runtime, train_text, test_text, train_score, test_score):
    print ("{0}: {1}".format(train_text, train_score))
    print ("{0}: {1}".format(test_text, test_score))
    # _df = pd.DataFrame({'Algorithm': [train_text.split('(')[1].replace(')', '')], 'Train Score': [train_score], 'Test Score': [test_score], 'Overfitting': [abs(float(train_score) - float(test_score))]})
    # pd.concat([df,_df])
    
    if '(' in train_text and ')' in train_text:
        result_dict['Algorithm'].extend([train_text.split('(')[1].replace(')', '')])
    else:
        result_dict['Algorithm'].extend(['Neural Networks'])
    result_dict['Time (s)'].extend([runtime])
    result_dict['Train Score'].extend([train_score])
    result_dict['Test Score'].extend([test_score])
    result_dict['Overfitting'].extend([abs(float(train_score) - float(test_score))])

def print_result():
    df = pd.DataFrame(result_dict)
    styles = ['plain', 'simple', 'github', 'grid', 'fancy_grid', 'pipe', 'orgtbl', 'jira', 'presto', 'pretty', 'psql', 'rst', 'mediawiki', 'moinmoin', 'youtrack', 'html', 'unsafehtml', 'latex', 'latex_raw', 'latex_booktabs', 'latex_longtable', 'textile', 'tsv']
    # for style in styles:
    #     print (style.upper())
    #     print(tabulate(df, headers = 'keys', tablefmt = style,  showindex="never"))
    print(tabulate(df, headers = 'keys', tablefmt = styles[4],  showindex="never"))
