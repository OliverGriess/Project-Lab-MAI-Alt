import sqlite3

# Connect to the database
conn = sqlite3.connect("metadata.db")

# Create a cursor object to interact with the database
cursor = conn.cursor()

# Query only the 'fake_indices' column from the 'metadata' table
cursor.execute("SELECT window_path, fake_indices FROM metadata")

# Fetch all rows from the 'fake_indices' column
rows = cursor.fetchall()
# Print the 'fake_indices' values

# Print the total number of rows
print(f"\nTotal number of rows in the 'metadata' table: {len(rows)}")

# Close the connection
conn.close()
