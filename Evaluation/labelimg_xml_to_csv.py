import os, sys
import glob
import pandas as pd
import xml.etree.ElementTree as ET

def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        base = os.path.basename(xml_file)
        realname = os.path.splitext(base)[0]+'.JPG'
        #realname = base
        print(realname)
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            bb = member.findall('bndbox')[0]
            value = (realname,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(bb.findall('xmin')[0].text),
                     int(bb.findall('ymin')[0].text),
                     int(bb.findall('xmax')[0].text),
                     int(bb.findall('ymax')[0].text),
                     member[3].text
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax', 'difficult']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main(inPath):
    inPath = "csiro_eval/"
    xml_df = xml_to_csv(inPath)
    xml_df.to_csv(inPath + '_xml_to_csv_labels.csv', index=None)
    print('Successfully converted xml to csv.')
   

if __name__ == "__main__":
    if (len(sys.argv) != 2):
        print('Usage: run.py <IN>')
        sys.exit(0)
    inPath = sys.argv[1]
    #inPath = "/mnt/omreast_users/phhale/csiro_trashnet/stormwater_vids/high_res_validation_set/"
    main(inPath)
