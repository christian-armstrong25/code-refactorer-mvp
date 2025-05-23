def get_min_max(data):
    # Function to find the minimum and maximum temperature from the data
    min_temp = 1000
    max_temp = -1000
    for row in data:
        temp = row[1]  # Doesn't convert to int, causing issues
        if temp < min_temp:
            min_temp = temp
        if temp > max_temp:
            max_temp = temp
    return min_temp, max_temp

def calculate_statistics():
    # Function to calculate and print the minimum and maximum temperatures
    data = [
        [1, 30, 50],
        [2, 28, 60],
        [3, 35, 55],
        # Missing dates and misformatted list items
        [5, 32],
        []
    ]
    min_temp, max_temp = get_min_max(data)
    print("Min Temp:", min_temp, "Max Temp:", max_temp)
    # Instead of CSV output, prints directly and doesn't handle None values well

calculate_statistics()