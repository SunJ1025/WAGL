import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def load_data(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    data = [line.strip().split() for line in lines]
    return data


def visualize_images(selected_data, query_folder, gallery_folder):
    fig, axes = plt.subplots(len(selected_data), 11, figsize=(20, 2 * len(selected_data)))

    for i, row in enumerate(selected_data):
        query_image = row[0]
        result_images = row[1:]

        # Display query image
        query_img_path = f"{query_folder}/{query_image}.jpeg"
        query_img = mpimg.imread(query_img_path)
        axes[i, 0].imshow(query_img)
        axes[i, 0].set_title('Query')
        axes[i, 0].axis('off')

        # Display result images
        for j, result_image in enumerate(result_images):
            result_img_path = f"{gallery_folder}/{result_image}.webp"
            result_img = mpimg.imread(result_img_path)
            axes[i, j + 1].imshow(result_img)
            axes[i, j + 1].set_title(f'Result {j + 1}')
            axes[i, j + 1].axis('off')

    plt.tight_layout()
    plt.show()


def main():
    with open("../acmmm2024/query_drone_name.txt", "r") as f:
        txt = f.readlines()
        f.close()
    txt = [i.split("\n")[0][:-5] for i in txt]
    query_names = txt
    filename = 'answer_no_graph_gem_bs_16_eph_60_dino_l_252_add_weather.txt'  # Replace with your txt file path
    query_folder = '/home/oem/桌面/drone/OneDrive_1_2024-5-11/query/query_drone160k_wx/query_drone_160k_wx_24'
    gallery_folder = '/home/oem/桌面/drone/OneDrive_1_2024-5-11/gallery/gallery_satellite_160k'  # Replace with your image folder path
    data = load_data(filename)

    for i in range(len(data)):
        data[i].insert(0, query_names[i])

    # Randomly select 10 lines
    print(data[0])
    selected_data = random.sample(data, 5)
    # selected_data = data[:10]
    print(selected_data)
    # Visualize selected images
    visualize_images(selected_data, query_folder, gallery_folder)


if __name__ == "__main__":
    main()
