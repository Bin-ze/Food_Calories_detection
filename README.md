We implemented **food detection** using mmdetection,We hope to help people eat healthier food by detecting food calories
Our model has achieved very good accuracy on the data set we built ourselves.

Food_class:

('pingguo', 'xiangjiao', 'fanqie', 'huanggua', 'xigua', 'li', 'juzi',
               'caomei', 'putao', 'mihoutao' )

See mmdetection official website for **environment configuration**: 
                
https://github.com/open-mmlab/mmdetection


Food_dataset:i

            链接：https://pan.baidu.com/s/1w9uGNZbi0rQnMLuid8ngAw 
            提取码：9ve2 
            --来自百度网盘超级会员V4的分享

train_weight:

            链接：https://pan.baidu.com/s/1VyA1xVu8wQcQ8bZIgGnC2Q 
            提取码：bslv 
            --来自百度网盘超级会员V4的分享

Test existing models:
            # single-gpu testing
            python tools/test.py \
                ${CONFIG_FILE} \
                ${CHECKPOINT_FILE} \
                [--out ${RESULT_FILE}] \
                [--eval ${EVAL_METRICS}] \
                [--show]

# multi-gpu testing

            bash tools/dist_test.sh \
                ${CONFIG_FILE} \
                ${CHECKPOINT_FILE} \
                ${GPU_NUM} \
                [--out ${RESULT_FILE}] \
                [--eval ${EVAL_METRICS}]
    

Training on a single GPU:

            python tools/train.py \
                ${CONFIG_FILE} \
                [optional arguments]
            CONFIG_FILE='../configs/pascal_voc/cascade_food_voc.py'

Training on multiple GPUs

            bash ./tools/dist_train.sh \
                ${CONFIG_FILE} \
                ${GPU_NUM} \
                [optional arguments]
            CONFIG_FILE='../configs/pascal_voc/cascade_food_voc.py'


    Result as following:

                ---------------iou_thr: 0.5---------------

                +-----------+-----+------+--------+-------+
                | class     | gts | dets | recall | ap    |
                +-----------+-----+------+--------+-------+
                | pingguo   | 408 | 419  | 0.995  | 0.995 |
                | xiangjiao | 106 | 114  | 0.991  | 0.991 |
                | fanqie    | 455 | 474  | 0.985  | 0.985 |
                | huanggua  | 225 | 217  | 0.840  | 0.839 |
                | xigua     | 252 | 268  | 0.980  | 0.980 |
                | li        | 349 | 360  | 0.966  | 0.965 |
                | juzi      | 303 | 314  | 0.980  | 0.980 |
                | caomei    | 537 | 559  | 0.978  | 0.977 |
                | putao     | 192 | 190  | 0.948  | 0.948 |
                | mihoutao  | 386 | 406  | 0.995  | 0.994 |
                +-----------+-----+------+--------+-------+
                | mAP       |     |      |        | 0.965 |
                +-----------+-----+------+--------+-------+
                OrderedDict([('AP50', 0.965), ('mAP', 0.9654127359390259)])