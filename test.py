import csv

# Read the CSV data into a list of rows
with open('flipkart_review_data_2022_02.csv', 'r' , encoding="utf8") as f:
    reader = csv.reader(f)
    rows = list(reader)

# Add a new column to each row
for row in rows:
    row.append('new value')

# Write the updated rows back to a CSV file
with open('output.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(rows)