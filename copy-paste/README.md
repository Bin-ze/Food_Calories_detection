We use copy-paste to generate fake data

We have annotated 380 pieces of data of 10 categories, and generated more data through this method for training the network.
Got very high accuracy.

**Strat:**

        run  python  example.py

will Generate as many images as you want.






Our data set is labeled as coco instance segmentation label,\
In this way, we can randomly take out the mask of the instance
and paste it on other images. In addition, we also randomly generate the position and size of the mask.
This greatly enhances the diversity of the data set, and then generates **label.txt**

**Format:**

        ../copy-paste-aug/fake_image/0.jpg 152,203,273,330,6 266,192,375,334,6

        format:
            file name   x_min,y_min,x_max,y_max,class_id


Then:

        run  python txt_xml.py 
            
will generate xml annotations.

Please integrate the generated data set into the following form

        -Food_voc
            -Annotatations  (xml)
            -JPEGImages    (image)

Finally: 

        run python   voc_split_trainval.py 

to generate a data set similar to the VOC data set catalog.

        -Food_voc
            --Annotatations  (xml)
                    1.xml
                    2.xml
                    ...
                    
            --ImageSets
                    ---Main
                        test.txt
                        train.txt
                        tarinval.txt
                        val.txt
            --JPEGImages    (image)
                    1.jpg
                    2.jpg
                    ...
            

At this point, the data preparation work is over, 
and the fake_Food data we need is generated.

