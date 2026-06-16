import xml.etree.ElementTree as ET
import math


def split_reviews_xml(input_filename, output_dir, chunk_size=35):
    try:
        tree = ET.parse(input_filename)
        root = tree.getroot()
    except FileNotFoundError:
        print(f"Error: The file {input_filename} was not found.")
        return
    except ET.ParseError:
        print("Error: The file is not a well-formed XML.")
        return

    # Extract all the <Review> elements
    reviews = root.findall("Review")
    total_reviews = len(reviews)

    if total_reviews == 0:
        print("No <Review> elements found in the XML.")
        return

    # Calculate how many files we will create
    total_parts = math.ceil(total_reviews / chunk_size)
    print(f"Found {total_reviews} reviews. Splitting into {total_parts} parts...\n")

    for i in range(total_parts):
        # Create a new root element <Reviews> for each file
        new_root = ET.Element("Reviews")

        # Determine the start and end indices for the current chunk
        start_index = i * chunk_size
        end_index = start_index + chunk_size
        chunk = reviews[start_index:end_index]

        # Append the chunked <Review> elements to the new root
        for review in chunk:
            new_root.append(review)

        # Create a new tree from the new root
        new_tree = ET.ElementTree(new_root)

        # Format the output filename (part1.xml, part2.xml, etc.)
        output_filename = f"{output_dir}/part{i + 1}.xml"

        # Write the tree to the new file, preserving UTF-8 encoding and the XML declaration
        new_tree.write(output_filename, encoding="UTF-8", xml_declaration=True)

        print(f"Created {output_filename} containing {len(chunk)} reviews.")


if __name__ == "__main__":
    split_reviews_xml("data/ABSA16_Restaurants_Train_SB1_v2.xml", "data")
