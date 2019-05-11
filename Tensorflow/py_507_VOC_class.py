voc = VOC()
yolo = YOLO(os.path.abspath(config["cls_list"]))

flag, data = voc.parse(config["label"])

if flag == True:

    flag, data = yolo.generate(data)
    if flag == True:
        flag, data = yolo.save(data, config["output_path"], config["img_path"] ,
                               config["img_type"], config["manipast_path"])

        if flag == False:
            print("Saving Result : {}, msg : {}".format(flag, data))

    else:
        print("YOLO Generating Result : {}, msg : {}".format(flag, data))


else:
    print("VOC Parsing Result : {}, msg : {}".format(flag, data))