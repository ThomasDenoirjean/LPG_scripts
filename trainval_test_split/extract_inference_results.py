import xml.etree.ElementTree as ET
import pandas as pd
import os


##### TODO : extraire toutes les 'proba' du modèles pour faciliter l'évaluation du modèle en faisant varier le threshold ?? ########

def extract_max_classification(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    result = {}
    for image in root.findall('images/image'):
        guid = image.find('.//GUID').text
        classifications = image.find('.//classifications')
        max_value = -1
        max_code = None
        for classification in classifications.findall('classification'):
            code = classification.find('code').text
            value = float(classification.find('value').text)
            if value > max_value:
                max_value = value
                max_code = code
        if max_code is not None:
            result[guid] = {'code': max_code, 'score': max_value}
    return result

result = extract_max_classification('test.xml')

df = pd.read_csv('test_labels.csv')

# Mapper le code de classification
df['pred_label'] = df['picture_guid'].map(lambda guid: result.get(guid, {}).get('code'))

# Mapper le score
df['pred_score'] = df['picture_guid'].map(lambda guid: result.get(guid, {}).get('score'))

if os.path.exists('test_labels_w_pred.csv'):
    os.remove('test_labels_w_pred.csv')

df.to_csv('test_labels_w_pred.csv', index=False)

print('Inference results have been extracted, now run python get_test_metrics.py')