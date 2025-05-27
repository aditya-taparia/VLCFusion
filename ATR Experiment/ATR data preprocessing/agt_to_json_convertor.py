import re
import json
import os
import argparse

def parse_text_to_json(text):
    """
    Parses a custom text format into a JSON-like dictionary structure.

    The text format consists of nested objects, attributes, keywords,
    multi-value lines, and time-value lines.

    Args:
        text (str): The input text to parse.

    Returns:
        dict: A dictionary representing the parsed structure.
    """
    lines = text.strip().split("\n")
    stack = [{}]  # Stack to manage nesting of objects
    
    # Regular expressions to match different line types
    object_start = re.compile(r"^\s*(\w+)\s*$")  # Matches lines like "ObjectName"
    attribute = re.compile(r'^\s*(\w+)\s*"([^"]+)"$')  # Matches lines like 'Key "Value"'
    keyword = re.compile(r'^\s*Keyword\s*"([^"]+)\s+([^"]+)"$') # Matches 'Keyword "Key Value"' (Note: original regex had a potential issue here, assuming Key and Value are separated by one or more spaces within quotes)
    multi_value = re.compile(r'^\s*(\w+)\s*([\d\.]+)\s*([\d\.]+)\s*$') # Matches 'Key 1.0 2.0'
    time_value = re.compile(r'^\s*(\w+)\s*((?:\d+\s*)+)$') # Matches 'Key 1 2 3 4'

    current_object = stack[-1] # The current dictionary being populated

    for line in lines:
        line = line.strip() # Remove leading/trailing whitespace from the line
        if not line: # Skip empty lines
            continue

        if line == "}":
            if len(stack) > 1: # Ensure we don't pop the root object
                stack.pop()
                current_object = stack[-1]
            else:
                # Handle potential error: unexpected closing brace
                print(f"Warning: Unexpected closing brace '}}' encountered at top level.")
            continue
        
        # Try to match an object start
        start_match = object_start.match(line)
        if start_match:
            new_object = {}
            object_name = start_match.group(1)
            
            # If the object_name already exists and is not a list, convert it to a list
            if object_name not in current_object:
                current_object[object_name] = new_object
            else:
                if not isinstance(current_object[object_name], list):
                    current_object[object_name] = [current_object[object_name]]
                current_object[object_name].append(new_object)
            
            stack.append(new_object)
            current_object = new_object
            continue

        # Try to match an attribute
        attr_match = attribute.match(line)
        if attr_match:
            key, value = attr_match.groups()
            # If the key already exists and is not a list, convert it to a list
            if key in current_object:
                if not isinstance(current_object[key], list):
                    current_object[key] = [current_object[key]]
                current_object[key].append(value)
            else:
                current_object[key] = value
            continue
        
        # Try to match a keyword line
        # Note: The original regex for keyword might need adjustment if "Key Value" can have multiple spaces.
        # This version assumes "Key" and "Value" are separated by one or more spaces within the quotes.
        key_match = keyword.match(line)
        if key_match:
            # The keyword regex captures "Key" and "Value" as two separate groups.
            # We'll store them as a dictionary or a structured way if needed,
            # or simply concatenate if that's the desired outcome.
            # For now, let's assume the first part is the key and the second is the value.
            # If "Keyword" itself is the key, and the content is the value, the regex and logic need adjustment.
            # Based on the original code, it seems "Keyword" is a special type, and the first group is the actual key.
            key_from_keyword, value_from_keyword = key_match.groups()

            # Storing as { "ActualKey": "Value" } under the current object.
            # If "Keyword" should be the key in current_object, this needs to change.
            # Let's assume 'Keyword' lines add an attribute named after the first quoted string.
            actual_key = key_from_keyword # This was 'key' in the original code, which is confusing.
                                        # Let's assume the first quoted string is the key.

            if actual_key in current_object:
                if not isinstance(current_object[actual_key], list):
                    current_object[actual_key] = [current_object[actual_key]]
                current_object[actual_key].append(value_from_keyword)
            else:
                current_object[actual_key] = value_from_keyword
            continue
        
        # Try to match a multi-value line
        multi_match = multi_value.match(line)
        if multi_match:
            key, value1, value2 = multi_match.groups()
            current_object[key] = [float(value1), float(value2)]
            continue
        
        # Try to match a time-value line
        time_match = time_value.match(line)
        if time_match:
            key, values_str = time_match.groups()
            # Split the string of values and convert each to int
            current_object[key] = list(map(int, values_str.strip().split()))
            continue
        
        # If no pattern matched, it could be an error or an unhandled line type
        if line: # Avoid printing for blank lines that might have been missed
            print(f"Warning: Unrecognized line format: '{line}'")

    if len(stack) > 1:
        # This indicates unclosed objects
        print(f"Warning: {len(stack) - 1} unclosed object(s) at the end of parsing.")
        # You might want to decide how to handle this - e.g., raise an error or try to auto-close.
        # For now, it will return the root object with potentially incomplete nested structures.

    return stack[0] # Return the root dictionary

def read_file_and_convert_to_json(file_path):
    """
    Reads a file, parses its text content using parse_text_to_json,
    and returns the resulting Python dictionary.

    Args:
        file_path (str): The path to the input file.

    Returns:
        dict: The parsed JSON data as a Python dictionary.
              Returns None if the file cannot be read.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file: # Added encoding
            text_data = file.read()
        json_data = parse_text_to_json(text_data)
        return json_data
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error reading or parsing file {file_path}: {e}")
        return None

def main():
    """
    Main function to handle command-line arguments,
    process files, and save JSON output.
    """
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Convert text files from a specific format to JSON.")
    parser.add_argument("agt_path", type=str, help="Path to the input directory containing .agt (or other) files.")
    parser.add_argument("save_path", type=str, help="Path to the output directory where JSON files will be saved.")
    
    args = parser.parse_args()

    input_dir = args.agt_path
    output_dir = args.save_path

    # --- Ensure output directory exists ---
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory: {os.path.abspath(output_dir)}")
    except OSError as e:
        print(f"Error creating output directory {output_dir}: {e}")
        return # Exit if output directory cannot be created

    # --- Process files ---
    if not os.path.isdir(input_dir):
        print(f"Error: Input path '{input_dir}' is not a valid directory.")
        return

    print(f"Processing files from: {os.path.abspath(input_dir)}")

    file_processed_count = 0
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            # Construct the full path to the source file
            file_path = os.path.join(root, file)
            
            print(f"Processing file: {file_path}...")
            json_data = read_file_and_convert_to_json(file_path)

            if json_data is not None:
                # Determine the name for the output JSON file
                save_name = os.path.splitext(file)[0]
                # Construct the full path for the output JSON file
                # This places the JSON file directly in output_dir, not preserving subdirectories from input_dir.
                # If subdirectory preservation is needed, this path construction needs to be more complex.
                save_file_path = os.path.join(output_dir, f"{save_name}.json")
                
                try:
                    with open(save_file_path, 'w', encoding='utf-8') as save_f: # Added encoding
                        json.dump(json_data, save_f, indent=4)
                    print(f"Successfully converted and saved to: {save_file_path}")
                    file_processed_count += 1
                except IOError as e:
                    print(f"Error writing JSON to file {save_file_path}: {e}")
                except Exception as e:
                    print(f"An unexpected error occurred while writing {save_file_path}: {e}")
            else:
                print(f"Skipping file due to previous errors: {file_path}")
    
    print(f"\nProcessing complete. {file_processed_count} file(s) processed.")

if __name__ == "__main__":
    main()