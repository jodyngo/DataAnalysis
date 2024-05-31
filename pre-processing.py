import pandas as pd
import socket, json, csv, os
import concurrent.futures

'''
#######################################
Data Analysis Model Pre-processing for Integrated Power Generation Operating System
#######################################

### Use variable ###
▶ PATH = the path on which the data is located


### How to use ###

1. Function loadTagList
    ▶ Enter the path where the tag list is located and return it to the list.

2. Function fetchValues (tag name, start time, end time, interval)
    ▶ Connect to the PHD in real time to store data in a designated folder.
    
3. Function getCorr (tag name 1, tag name 2)
    ▶ Obtain and output the correlation coefficients for tag 1 and tag 2. If the data are not the same length, skip it.

4. Global Variables
    
    The folder where PATH = CSV is stored.
    FILE_NAME = File name to store correlation coefficients.
    runCount = Check the number of times a repeat statement is executed. Usually, no modifications are required.

'''

# Defining Global Variables
PATH = './hourData'
FILENAME = 'corr_result.csv'
runCount = 0 # Check the number of runs of #runCount repeat statements

def loadTagList(tagList):
    tagFile = open(tagList,'r')
    lines = tagFile.readline()
    tagFile.close()
    for i in range(len(lines)):
        lines[i] = lines[i].strip()
    return lines
def fetchValues(tagName, startTime, endTime, freq):
    reqFormat = '{"tagList":[{"tagName":"","startTime":"","endTime":"","timeType":"1","unit":"0","frequency":""}]}'
    req = json.loads(reqFormat)
    req['tagList'][0]['tagName'] = tagName
    req['tagList'][0]['startTime'] = startTime
    req['tagList'][0]['endTime'] = endTime
    req['tagList'][0]['freq'] = freq
    msg = json.dumps(req)

    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(("127.0.0.0.1", 8208))
    client.send(msg.encode("utf-8"))
    data = client.recv(1024*1024)
    tagValues = []

    for valueList in msg:
        values = valueList.get('values')
        for val in values:
            tagValues.append(val.get('value'))
    time = startTime.split(" ")[0]
    with open(f"./hourData/{tagName}.csv", mode='w', newline="") as file:
        for i in range(len(tagValues)):
            writer = csv.writer(file)
            writer.writerow(tagValues[i][0], tagValues[i][1])
    file.close()

# Calculation of correlation coefficients (Pearson)
def getCorr(valueList1, valueList2):
    try:
        df = pd.DataFrame({'X':valueList1, 'Y':valueList2})
        matrix = df.corr(method='pearson')
        return matrix.loc['X','Y']
    except ValueError as e:
        print("It should be the same length")
        pass

# Collect the names of all files in a folder and save them as a list
file_names = [f for f in os.listdir(PATH) if os.path.isfile(os.path.join(PATH, f))]

# Calculation of correlation coefficients for all file combinations
with concurrent.futures.ThreadPoolExecutor() as executor:
    for i in range(len(file_names)):
        for j in range(i + 1, len(file_names)):

            file1 = file_names[i]
            file2 = file_names[j]
            data1 = pd.read_csv(os.path.join(PATH, file1))
            data2 = pd.read_csv(os.path.join(PATH, file2))
            col1 = data1.iloc[:, 1]
            col2 = data2.iloc[:, 1]

            # 상관계수 계산 (피어슨)
            correlation = col1.corr(col2, method='pearson')

            # 결과 출력
            print(f"[{runCount}] {file1} vs {file2} = {correlation}")
            # 파일 열기
            with open('corr_result.csv', mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([file1.split('_')[0], file2.split('_')[0], correlation])
            runCount +=1