import csv

def get_detection_logs():
    rows = []
    with open("transaction.csv", 'r') as file:
        csvreader = csv.reader(file)
        header = next(csvreader)
        for row in csvreader:
            logs = {}
            if "UNKNOWN" not in row:
                for count, item in enumerate(row):
                    logs[header[count]] = item
                rows.append(logs)
    return rows

def get_transaction_logs():
    rows = []
    with open("transaction.csv", 'r') as file:
        csvreader = csv.reader(file)
        header = next(csvreader)
        for row in csvreader:
            logs = {}
            for count, item in enumerate(row):
                logs[header[count]] = item
            rows.append(logs)
    return rows

if __name__ == "__main__":
    get_transaction_logs()