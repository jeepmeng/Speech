import json

test_dict = {'dic_filename':'dict.txt',
             'dataset':{
                 'st-cmds'{
                 'train':{'data_list':r'/Users/liufucong/Downloads/ltxm/ContextNet-master/datalist/st-cmds/train.wav.txt',
                             'data_pth':r'/Users/liufucong/Downloads/ltxm/ST-CMDS-20170001_1-OS',
                             'label_list':r'/Users/liufucong/Downloads/ltxm/Speech/st-cmds/train.syllable.txt'},
                 'test':{},
                 'dev':{}
                         }
                     }
                 },
                 'test':{},
                 'dev':{},
             }
print(test_dict['dataset']['train']['st-cmds_train'])
print(type(test_dict))
#dumps 将数据转换成字符串
# json_str = json.dumps(test_dict)
# print(json_str)
# print(type(json_str))

with open("./config.json","w") as f:
     json.dump(test_dict,f)
     print("加载入文件完成...")