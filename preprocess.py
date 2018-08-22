from deeplab import *
from dataset import *
import io

dataset_dir = '/home/suka/eliceproject_dataset/lookbook2/data/'
models = []
clothes = []

for filename in os.listdir(dataset_dir):
    if filename.endswith('.jpg'):
        if filename.split('_')[1].endswith('0'):
            models.append(filename)

MODEL = DeepLabModel("./deeplabv3_mnv2_pascal_train_aug.tar.gz")     
for i in models:
    img = run_segmentation(MODEL,dataset_dir+i)
    img = Image.fromarray(img)
    buffer = io.BytesIO()
    img.save(buffer, format = "JPEG")
    open("/home/suka/eliceproject_dataset/lookbook3/data/"+i, "wb").write(buffer.getvalue())


# def main():
    
#     data = LookbookDataset(data_dir="/home/suka/eliceproject_dataset/lookbook2/data/",index_dir="/home/suka/PycharmProjects/pixelDTgan/")
#     images,clothes = data.getimage()
#     print(len(images),len(clothes))
    
# main()