from img_to_mnist import image_to_mnist_array,save_idx_images

arr=image_to_mnist_array("mydigit.png")
save_idx_images("output.idx3-ubyte",arr)