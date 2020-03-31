def RandomForestFeatureSelection(data,target = 'y',fliter_pct = 0.7):
    '''
    data为数据框，target为标签,filted_pct为筛选百分比，通过随机森林的方式确定变量，筛选出重要性靠前的变量
    '''
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    
    x = data.drop(columns=[target])
    y = data.y
    rfc = RandomForestClassifier(n_estimators=6,random_state=222)
    rfc1 = rfc.fit(x,y)
    importance_dt = pd.DataFrame({'cols':x.columns,'importance':rfc1.feature_importances_}).sort_values('importance',ascending=False)
    n_vars = int(len(x.columns)*filter_pct)
    important_vars = importance_dt.iloc[1:n_vars,0].tolist()
    print('important features are:\n',important_vars)
    data_filted = data.reindex(columns=[target]+important_vars).copy()
    
    return(data_filted)