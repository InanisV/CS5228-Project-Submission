import pandas as pd
import numpy as np
from sklearn import preprocessing

def generate_encoder(train_, test_, feature):
    train_df = train_[[feature]]
    test_df = test_[[feature]]
    make_list = train_df[feature].tolist()
    make_list.extend(test_df[feature].tolist())
    make_list = [str(i).lower() for i in make_list]
    make_list = np.array(make_list)
    make_encoder = preprocessing.LabelEncoder()
    make_encoder.fit(make_list)
    return make_encoder


ori_train = pd.read_csv("train.csv").drop_duplicates(subset=["listing_id"], keep="first")
ori_test = pd.read_csv("test.csv")
my_train = pd.read_csv("train_v6.csv")
my_test = pd.read_csv('test_v6.csv')

my_index_train = ["listing_id", "accessories_vectors", "years of warranty","better loan offer","well maintained","low fuel consumption","reg_date","power","engine_cap","mileage","no_of_owners","depreciation","coe","dereg_value","omv","arf","transmission","model_price", "is_new", "price"]
my_index_test = ["listing_id", "accessories_vectors", "years of warranty","better loan offer","well maintained","low fuel consumption","reg_date","power","engine_cap","mileage","no_of_owners","depreciation","coe","dereg_value","omv","arf","transmission","model_price", "is_new"]
my_train = my_train[my_index_train].drop_duplicates(subset=["listing_id"], keep="first")
my_test = my_test[my_index_test]

print(my_train.shape)
print(my_test.shape)
print(ori_test.shape)
#处理fuel_type
ori_train['fuel_type'] = ori_train['fuel_type'].map(lambda x: x if x is not np.nan else 'petrol')
ori_test['fuel_type'] = ori_test['fuel_type'].map(lambda x: x if x is not np.nan else 'petrol')
fuel_encoder = generate_encoder(ori_train, ori_test, "fuel_type")
ori_train["fuel_type"] = fuel_encoder.transform(list(ori_train['fuel_type'].values))
ori_test["fuel_type"] = fuel_encoder.transform(list(ori_test['fuel_type'].values))

#处理category
cat_encoder = generate_encoder(ori_train, ori_test, "category")
ori_train["category"] = cat_encoder.transform(list(ori_train['category'].values))
ori_test["category"] = cat_encoder.transform(list(ori_test['category'].values))

#处理type_of_vihecle
tov_encoder = generate_encoder(ori_train, ori_test, "type_of_vehicle")
ori_train["type_of_vehicle"] = tov_encoder.transform(list(ori_train['type_of_vehicle'].values))
ori_test["type_of_vehicle"] = tov_encoder.transform(list(ori_test['type_of_vehicle'].values))

#处理make
cat_encoder = generate_encoder(ori_train, ori_test, "make")
ori_train["make"] = cat_encoder.transform(list(ori_train['make'].values))
ori_test["make"] = cat_encoder.transform(list(ori_test['make'].values))

#处理make
cat_encoder = generate_encoder(ori_train, ori_test, "model")
ori_train["model"] = cat_encoder.transform(list(ori_train['model'].values))
ori_test["model"] = cat_encoder.transform(list(ori_test['model'].values))

#缩减ori dataset

ori_train = ori_train[["listing_id", "category", "type_of_vehicle", "fuel_type", "make", "model"]]
ori_test = ori_test[["listing_id", "category", "type_of_vehicle", "fuel_type", "make", "model"]]
res_train = pd.merge(my_train, ori_train, on="listing_id", how="inner")
my_test["category"] = ori_test["category"]
my_test["fuel_type"] = ori_test["fuel_type"]
my_test["type_of_vehicle"] = ori_test["type_of_vehicle"]
my_test["make"] = ori_test["make"]
my_test["model"] = ori_test["model"]
res_train.to_csv("train_v7_2.csv")
my_test.to_csv("test_v7_2.csv")
print(res_train.shape)
print(my_test.shape)

