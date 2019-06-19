import os


# Delete the image if it is not in data.txt
def delete2():
    image_path = '_out/'
    image_list = os.listdir(image_path)
    image_list = [image_path+i for i in image_list]

    file_images = []

    with open("data.txt", "r") as f:
        for line in f:
            file_images.append("_out/" + line.split()[0])

    for image in image_list:
        if image not in file_images:
            if image != "data.txt":
                os.remove(image)

# Delete the repeated values in data.txt
# Delete the image in data.txt if it is not exist
def delete():
    final = []
    control = []

    image_path = '_out/'
    image_list = os.listdir(image_path)
    image_list = [image_path+i for i in image_list]
    print(image_list)
    with open("data.txt", "r") as f:
        for line in f:
            elm = line.split()[0][:-4]
            print(elm)
            if elm in control:
            	a = 1
            	#print("elm: {} is in control".format(elm))
            else:
            	print("elm: {} is not in control".format(elm))
            	control.append(elm)
            	temp = "_out/" + line.split()[0]
            	print(temp)
            	if temp in image_list:
            		#print("elm is in image_list")
            		final.append(line)


    with open("data.txt", "w") as w:
    	for line in final:
    		w.write(line)

def normalize():

    with open("data.txt", "r") as f:
        lines = f.readlines()


    out = open("data.txt", "w")
    for line in lines:
        new_val = "{0:.6f}".format(float(line.split()[1]) * 180)
        new_line = line.split()[0] + " " + new_val + "\n"
        out.write(new_line)

    print("Normalized operation over")
    
if __name__ == '__main__':
    delete()
    normalize()
    delete2()
