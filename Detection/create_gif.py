import os, sys
from PIL import Image

def gif_creator(folder_name ='frcnn_ds2s_Syn0Real100_111720_step20k_Valv0'):
    print("creating GIF")
    gif_images = []
    image_folder = f'../test_images/{folder_name}/bbox/'
    gif_frames = os.listdir(image_folder)
    for n in gif_frames:
        frame = Image.open(os.path.join(image_folder,n))
        gif_images.append(frame)

    # Save the frames as an animated GIF
    gif_images[0].save(f'../test_images/{folder_name}.gif',
                save_all=True,
                append_images=gif_images[1:],
                duration=1500,
                loop=5)
    print("GIF created: ", f'../test_images/{folder_name}.gif')

    
if __name__ == "__main__":
    gif_creator()