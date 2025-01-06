import csv

total = 0

with open("data/coffee_heaven_sales.csv", 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        total += float(row['TotalAmount'])

print(f"Total sales: {total}")