def merge_sort(arr):
    if len(arr) > 1:
        mid = len(arr) // 2  # Find the middle of the array
        left_half = arr[:mid]  # Dividing the elements into 2 halves
        right_half = arr[mid:]

        merge_sort(left_half)  # Sorting the first half
        merge_sort(right_half)  # Sorting the second half

        i = j = k = 0

        # Copy data to temp arrays L[] and R[]
        while i < len(left_half) and j < len(right_half):
            if left_half[i] < right_half[j]:
                arr[k] = left_half[i]
                i += 1
            else:
                arr[k] = right_half[j]
                j += 1
            k += 1

        # Checking if any element was left
        while i < len(left_half):
            arr[k] = left_half[i]
            i += 1
            k += 1

        while j < len(right_half):
            arr[k] = right_half[j]
            j += 1
            k += 1

    return arr

# deez monumental nuts

def read_and_sort(input_file, output_file):
    with open(input_file, 'r') as file:
        numbers = file.read().splitlines()

    # Convert strings to integers
    numbers = [int(num) for num in numbers]

    # Sort the numbers using merge sort
    sorted_numbers = merge_sort(numbers)

    # Write the sorted numbers to an output file
    with open(output_file, 'w') as file:
        for num in sorted_numbers:
            file.write(f"{num}\n")

    print(f"Sorted numbers have been written to {output_file}")


if __name__ == "__main__":
    input_file = "input_numbers.txt"
    output_file = "sorted_numbers.txt"
    read_and_sort(input_file, output_file)
