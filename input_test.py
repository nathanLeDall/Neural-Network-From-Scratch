# Ask the user for the file path
file_path = "data/L30fft_32.out"

try:
    with open(file_path, 'r', encoding='utf-8') as f:
        # Read first line for the two numbers
        first_line = f.readline().strip()
        size_numbers = [int(x) for x in first_line.split()]
        print("First line numbers:", size_numbers)

        # Prepare storage for labels and data
        labels = []
        data = []

        # Read the rest of the lines
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue  # skip empty or invalid lines
            label = int(parts[0])         # first number is the label
            features = [float(x) for x in parts[1:]]  # rest are data
            labels.append(label)
            data.append(features)

    print("Labels:", labels)
    print("Data:", data)

except FileNotFoundError:
    print("File not found. Please check the path.")
except Exception as e:
    print(f"An error occurred: {e}")

