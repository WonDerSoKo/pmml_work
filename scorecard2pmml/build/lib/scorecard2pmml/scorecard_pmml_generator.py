"""PMML Scorecard generator"""

import os.path
import string
from lxml import etree
import tabulate
import numpy as np
import pandas as pd
import re
import json
import time


# developed by shiming.ren@tcredit.com on Nov.23rd 2019

def func_json_gen(bin_tbl,preproc_tbl,path='/Users/simon/Downloads/',model_name="hahaha1"):
    '''
    version:1.0.0
    bin_tbl 是分箱表 包含三个字段 variable bin points 第一行是基础分
    preproc_tbl 是预处理表 包含四个字段 Var_name Impute_value Center Scale
    '''
    import numpy as np
    import pandas as pd
    import re
    import json
    # 获取最大最小分
    initial_score = bin_tbl[bin_tbl['variable']=="basepoints"]['points'].tolist()[0]
    var_pt = bin_tbl[1:].reindex(columns=['variable','points'])
    min_score = var_pt.groupby('variable').agg(['max','min']).sum(axis=0)['points']['min']+initial_score
    max_score = var_pt.groupby('variable').agg(['max','min']).sum(axis=0)['points']['max']+initial_score
    # 获取变量名
    x = bin_tbl[1:].copy()
    x.reset_index(inplace=True)
    x['intvl'] = x['bin'].str.strip('\[|\)').str.split(',').apply(lambda x:[">= "+x[0],"< "+x[1]])\
        .apply(lambda x:x[1] if re.search('Inf',x[0]) else x)\
        .apply(lambda x:x[0] if re.search('Inf',x[1]) else x)
    x['json'] = [{'partialScore':j,'predicate':i} for i,j in zip(x['intvl'],x['points'])]
    y = x.reindex(columns=['variable','json']).groupby('variable').agg(lambda x:list(x))['json']
    z = pd.DataFrame({'Var_name':y.index,'Json':y.values})
    # json串
    var_info = y.values
    var_nm = y.index
    #转换
    preproc_tbl['Ref_1'] = preproc_tbl['Center']
    preproc_tbl['Ref_2'] = preproc_tbl['Center'] + preproc_tbl['Scale']
    # 结果
    res = {}
    res['model_name'] = model_name
    res['point_range'] = [{'initial_score':str(initial_score),'min_score':str(min_score),'max_score':str(max_score)}]
    res['data_fields'] = [{'name':i,'dataType':'double','optype':'continuous'} for i in var_nm]
    res['characteristics'] = [{"name": i,"baselineScore": 0,
                               "impute":str(preproc_tbl[preproc_tbl["Var_name"]==i]["Impute_value"].values[0]),
                              "ref":[str(preproc_tbl[preproc_tbl["Var_name"]==i]["Ref_1"].values[0]),
                                     str(preproc_tbl[preproc_tbl["Var_name"]==i]["Ref_2"].values[0])],
                               "attributes":j
                               } for i,j in zip(var_nm,var_info)]
    # with open(path+model_name+'.json','w') as f:
    #     f.write(json.dumps(res,indent=4,separators=(',', ':')))
    # return(json.dumps(res,indent=4,separators=(',', ':')))
    return(res)

def func_pmml_gen(input_spec,path="/Users/simon/Downloads/",copyright="Tcredit",description="write your model description here."):
    '''
    input_spec:
    json result from function func_json_gen
    path:
    model saving path
    copyright:
    information from developer
    description:
    your model description
    '''
    model_name = input_spec['model_name']
    # Header
    root = etree.Element("PMML", version="4.2",
                         xmlns="http://www.dmg.org/PMML-4_2")
    header = etree.SubElement(root, "Header",copyright=copyright,
                             description=description)
    # DataDictionary input的数据字典包括字段名，类型
    datadict = etree.SubElement(root, "DataDictionary")

    for fea_info in input_spec['data_fields']:
        name,dataType,optype = fea_info.values()
        field = etree.SubElement(datadict, "DataField", name=name,
                             dataType=dataType, optype=optype)

    # scorecard
    scorecard = etree.SubElement(root, "Scorecard",
                             modelName=model_name,
                             functionName="regression",
                             useReasonCodes="false",
                             reasonCodeAlgorithm="pointsAbove",
                             initialScore=str(input_spec['point_range'][0]['initial_score']),
                             baselineScore="1",
                             baselineMethod="min")


    # LocalTransformations 预处理标准化
    local_tran = etree.SubElement(scorecard, "LocalTransformations")

    for i,j in zip(input_spec['data_fields'],input_spec['characteristics']):
    # print(j['name'])
    # print(i['optype'])
    # print(i['dataType'])
    # print(j['ref'])
    # print(j['impute'])
        drv_field = etree.SubElement(local_tran, "DerivedField",
                                    name=i['name']+'_drv',optype=i['optype'],dataType=i['dataType'])
        normcon = etree.SubElement(drv_field,"NormContinuous",field=i['name'])
        linearnorm1 = etree.SubElement(normcon,"LinearNorm",orig=(j['ref'][0]),norm="0")
        linearnorm2 = etree.SubElement(normcon,"LinearNorm",orig=(j['ref'][1]),norm="1")

    # MiningSchema 入模变量名称 包括字段名，缺失填充值等信息
    schema = etree.SubElement(scorecard, "MiningSchema")

    for j in input_spec['characteristics']:
        # print(j['name'])
        # print(j['ref'])
        # print(j['impute'])
        element = etree.SubElement(schema, "MiningField", name=j['name'],invalidValueTreatment="asMissing",
            missingValueReplacement=str(j['impute']),missingValueTreatment="asValue")

    # Output 输出字段名，类型
    output = etree.SubElement(scorecard, "Output")
    etree.SubElement(output, "OutputField",
                     name="RiskScore",
                     feature="predictedValue",
                     dataType="double",
                     optype="continuous")

    sc_ref = etree.SubElement(output, "OutputField",
                name="FinalRiskScore",
                feature="transformedValue",
                dataType="string",
                optype="continuous")

    # 分数映射 
    score_range = [input_spec['point_range'][0]['min_score'],input_spec['point_range'][0]['max_score']]


    sc_fl = etree.SubElement(sc_ref,"Apply",
                function="floor")
    sc_nc = etree.SubElement(sc_fl,"NormContinuous",
                    field = "RiskScore")

    etree.SubElement(sc_nc,"LinearNorm",
                    orig=str(score_range[0]),norm="350"
                    )

    etree.SubElement(sc_nc,"LinearNorm",
                orig=str(score_range[1]),norm="950"
                )

    # Characteristics 字段分箱
    characteristics = etree.SubElement(scorecard, "Characteristics")

    #  区间对应关系表
    _cmpopmap = {
    "<":"lessThan",
    "<=":"lessOrEqual",
    "==":"equal",
    ">=":"greaterOrEqual",
    ">":"greaterThan",
    }
    # _cmpopmap

    for chrstc in input_spec['characteristics']:
        characteristic = etree.SubElement(characteristics, "Characteristic",
                                      name = chrstc['name'] + "_score")
        for attribute in chrstc['attributes']:
    #         print(attribute)
            score,intvl = attribute['partialScore'],attribute['predicate']
    #         print(score,intvl)

            attrib = etree.SubElement(characteristic, "Attribute",
                                         partialScore = str(score))

            if type(intvl)==list:
                optr = [_cmpopmap.get(symb.split(' ')[0].strip()) for symb in attribute['predicate']]
                val = [symb.split(' ')[1].strip() for symb in attribute['predicate']]
                com_pred = etree.SubElement(attrib,"CompoundPredicate",
                                           booleanOperator="and")
                sim_pred1 = etree.SubElement(com_pred,"SimplePredicate",
                                            field=chrstc['name']+'_drv',operator=optr[0],value=str(val[0]))
                sim_pred2 = etree.SubElement(com_pred,"SimplePredicate",
                                            field=chrstc['name']+'_drv',operator=optr[1],value=str(val[1]))
            else:
                optr = _cmpopmap.get(attribute['predicate'].split(' ')[0].strip())
                val = attribute['predicate'].split(' ')[1].strip()
                sim_pred = etree.SubElement(attrib,"SimplePredicate",
                                           field=chrstc['name']+'_drv',operator=optr,value=str(val))


    scorecard_xml = etree.tostring(root, pretty_print=True, xml_declaration=True,encoding='utf-8')

    with open(path+model_name+".pmml",'wb') as f:
        f.write(scorecard_xml)
    print(path+model_name+".pmml"+' saved at '+time.ctime())
    return(scorecard_xml)

def characteristic_table(input_spec):
    """
    result from function func_json_gen
    """
    headings = "Criterion", "Partial Score"
    data_rows = []
    for fea in input_spec['characteristics']:
        name = fea['name']
        attributes = fea['attributes']
        data_rows.append(("===== " + name + " =====", "===="))
        for score_nd_pred in attributes:
            partialScore = score_nd_pred['partialScore']
            predicate = score_nd_pred['predicate']
            if predicate is None:
                full_predicate = name + " missing"
            else:
                try:
                    full_predicate = name + " " + predicate
                except TypeError:
                    predicate_parts = [name + " " + part for part in predicate]
                    full_predicate = " && ".join(predicate_parts)
            data_rows.append((full_predicate, partialScore))
    print(tabulate.tabulate(data_rows, headings, tablefmt="psql"))



def scorecard2pmml(bin_tbl,preproc_tbl,path='/Users/simon/Downloads/',model_name="hahaha1",copyright="Tcredit",description="write your model description here."):
    input_s = func_json_gen(bin_tbl,preproc_tbl,path=path,model_name=model_name)
    characteristic_table(input_s)
    pmml_str = func_pmml_gen(input_s,path=path)





