import csv
import numpy as np


def load_csv():
    # Initialize an empty list
    data = []

    # Open the CSV file with the `csv` module
    with open('q-charts.csv', 'r') as f:
        # Create a CSV reader
        reader = csv.reader(f)

        # Iterate over the rows of the CSV file
        for row in reader:
            # Append the row to the list
            data.append(row)

    print(data)
    # Output: [['row_number', 'data1', 'data2', 'data3', 'data4'],
    # ['1', '0.2540', '0.6350', '0.3405', '...'],
    # ['2', '0.0941', '0.8449', '0.5678', '...'], ..., ['300', '0.3412', '0.5678', '0.2345', '...']]

    # delete the first element(row_number) in each row:
    for row in data:
        row.pop(0)

    # convert string to float:
    count = 0
    for row in data:
        count += 1
        for i in range(len(row)):
            row[i] = -float(row[i])

    return np.array(data)

    # load_csv()
