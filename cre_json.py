import json
import yaml

test_dict = \
    {'dic_filename':'dict.txt',
     'st-cmds':{
                 'train':{'data_list':r'/Users/liufucong/Downloads/ltxm/ContextNet-master/datalist/st-cmds/train.wav.txt',
                             'data_pth':r'/Users/liufucong/Downloads/ltxm/ST-CMDS-20170001_1-OS',
                             'label_list':r'/Users/liufucong/Downloads/ltxm/Speech/st-cmds/train.syllable.txt'}

                 },
            'second_dataset': r'data.txt'

             }
# print(test_dict['st-cmds']['train'])
# print(type(test_dict))
# print(test_dict)



#dumps 将数据转换成字符串
# json_str = json.dumps(test_dict)
# print(json_str)
# print(type(json_str))

# with open("./config.json","w") as f:
#      json.dump(test_dict,f)
#      print("加载入文件完成...")

# data1: 123
# data2:
#   k1: v1
#   k2:
#   - 4
#   - 5
#   - 6


with open("config_win.yml", "r", encoding="utf8") as f:
    a = yaml.load(f, Loader=yaml.FullLoader)

    print(a)