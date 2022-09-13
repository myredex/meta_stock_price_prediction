# Meta Inc. Stock price predictions
In this jupiter notebook I'll try to predict stocks price using webscrapping and multiple linear regression model.

Let's take a look to stock prices and market capitalisation as a result of how market reacts to financial statistics of the company. 
First of all we need to extract financial statistics from the web. After that we need to determine affection of this statistics to stock price and discover correlation.
Final step will be building model to predict market capitalisation for the next quorter.

Import libraries


```python
# Install if required
#!mamba install bs4==4.10.0 -y
#!mamba install html5lib==1.1 -y
```


```python
#Import libraries for webscrapping
import pandas as pd
import requests
from bs4 import BeautifulSoup
```

Let's create function that will help us to extract data from macrotrends.net website. 


```python
def get_dataframe_from_macrotrends_net(url, list_of_columns, table_number=0, column_index=1):
    # This function helps to get information from Macrotrends.net using simple navigation
    # You only need to enter url, list of parced columns in the table and 
    # number of the table (first, second ets in int format)

    html_data = requests.get(url).text   #get data from the webpage
    souped_html = BeautifulSoup(html_data, 'html5lib') #Use beautifulsoup

    # Find the second table with Quarterly Revenue /or use 0 for first table
    parced_table = souped_html.find_all('table')[table_number]   

    #Create blank dataframe 
    dataframe = pd.DataFrame(columns=list_of_columns) 
 
    # Find all tr tags and put data into Dataframe
    non_float_values = 0
    for row in parced_table.find("tbody").find_all("tr"):  

        col = row.find_all("td") 
        date = col[0].text 
        col1 = col[column_index].text
        col1 = col1.replace('$', '').replace(',', '')

        if col1.isdigit():
            col1 = float(col1)
            dataframe = dataframe.append({list_of_columns[0]:date, list_of_columns[1]:col1}, ignore_index=True)
        else: 
            dataframe = dataframe.append({list_of_columns[0]:date, list_of_columns[1]:col1}, ignore_index=True)
            non_float_values = non_float_values + 1
    
    print("Non float values:", non_float_values) #How many rows were missed
    return dataframe
```

Let's collect data: Revenue, Gross profit etc.


```python
base_url = "https://www.macrotrends.net/stocks/charts/META/meta-platforms"
# Set list of tupples to parce
url_list = [(base_url + "/revenue", ["Date", "Revenue"], 1, 1),
            (base_url + "/gross-profit", ["Date", "Gross profit"], 1, 1),
            (base_url + "/eps-earnings-per-share-diluted", ["Date", "EPS"], 1, 1),
            (base_url + "/ebitda", ["Date", "EBITDA"], 1, 1),
            (base_url + "/operating-income", ["Date", "Operating income"], 1, 1),
            (base_url + "/pe-ratio", ["Date", "PE ratio"], 0, 3),
            (base_url + "/net-income", ["Date", "Net income"], 1, 1)
            ]

i=0
# Get dfs and join them to one dataframe
for url, list_of_cols, table_num, column_index in url_list: 
    #Get dataframe
    dataframe = get_dataframe_from_macrotrends_net(url, list_of_cols, table_num, column_index)
    
    # During first iterration resut df equals to dataframe
    if i == 0:
        result_df = dataframe
        
    # Second and other iterration we merge result df and dataframe we got with function
    else:
        result_df = result_df.merge(dataframe, how='left')
        
    i=i+1
    
revenue_df = result_df
revenue_df.head()
```

    Non float values: 1
    Non float values: 1
    Non float values: 51
    Non float values: 2
    Non float values: 2
    Non float values: 39
    Non float values: 3





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Revenue</th>
      <th>Gross profit</th>
      <th>EPS</th>
      <th>EBITDA</th>
      <th>Operating income</th>
      <th>PE ratio</th>
      <th>Net income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2022-06-30</td>
      <td>28822</td>
      <td>23630</td>
      <td>2.46</td>
      <td>10337</td>
      <td>8358</td>
      <td>13.36</td>
      <td>6687</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2022-03-31</td>
      <td>27908</td>
      <td>21903</td>
      <td>2.72</td>
      <td>10680</td>
      <td>8524</td>
      <td>16.82</td>
      <td>7465</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2021-12-31</td>
      <td>33671</td>
      <td>27323</td>
      <td>3.64</td>
      <td>14599</td>
      <td>12585</td>
      <td>24.37</td>
      <td>10285</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2021-09-30</td>
      <td>29010</td>
      <td>23239</td>
      <td>3.22</td>
      <td>12418</td>
      <td>10423</td>
      <td>24.22</td>
      <td>9194</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2021-06-30</td>
      <td>29077</td>
      <td>23678</td>
      <td>3.61</td>
      <td>14353</td>
      <td>12367</td>
      <td>25.76</td>
      <td>10394</td>
    </tr>
  </tbody>
</table>
</div>




```python
revenue_df.shape
```




    (51, 8)




```python
revenue_df.dtypes
```




    Date                object
    Revenue             object
    Gross profit        object
    EPS                 object
    EBITDA              object
    Operating income    object
    PE ratio            object
    Net income          object
    dtype: object



Now when revenue data extracted we need stocks price data.

## Extracting stock prices

I'm going to use Yahoo finance API to extract stock prices history.


```python
# Import library
import yfinance as yf

# Set ticker
meta = yf.Ticker("META")

# Get history data
hist = meta.history(period="max")

hist.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Volume</th>
      <th>Dividends</th>
      <th>Stock Splits</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2012-05-18</th>
      <td>42.049999</td>
      <td>45.000000</td>
      <td>38.000000</td>
      <td>38.230000</td>
      <td>573576400</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2012-05-21</th>
      <td>36.529999</td>
      <td>36.660000</td>
      <td>33.000000</td>
      <td>34.029999</td>
      <td>168192700</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2012-05-22</th>
      <td>32.610001</td>
      <td>33.590000</td>
      <td>30.940001</td>
      <td>31.000000</td>
      <td>101786600</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2012-05-23</th>
      <td>31.370001</td>
      <td>32.500000</td>
      <td>31.360001</td>
      <td>32.000000</td>
      <td>73600000</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2012-05-24</th>
      <td>32.950001</td>
      <td>33.209999</td>
      <td>31.770000</td>
      <td>33.029999</td>
      <td>50237200</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Let's visualise data


```python
# Reset index
hist.reset_index(inplace=True)
# Plot the chart
hist.plot(x="Date", y="Close")
```




    <AxesSubplot:xlabel='Date'>




    
![png](output_16_1.png)
    


Now we need to join two dataframes. For this purposes I'll use this function


```python
def join_price_and_revenue_dfs(price_df, revenue_df, shift=0):
    # Returns new dataframe after comparison of the dates. 

    # Import required libraries
    from datetime import datetime
    from datetime import timedelta
    
    # Set last day var to determine since what day stock prices is availiable
    last_history_date = hist.iloc[0,0]
    # Set time delta
    timedelta_object = timedelta(days=1)
    # Create blank list to collect prices from second df
    price_list =[]
    
    # Loop trought Revenue dataframe. 
    d1 = 0
    d2 = 0
    d3 = 0
    d4 = 0
        
    for index, row in revenue_df.iterrows():
        
        # Get date of the report. Then to exclude affection of the reports to the data
        # I'll use dates two days, three days and for days before the report.
        revenue_date = datetime.strptime(revenue_df.iloc[index, 0], "%Y-%m-%d")
        two_days_before_revenue_date = revenue_date - timedelta_object * 2
        three_days_before_revenue_date = revenue_date - timedelta_object * 3
        four_days_before_revenue_date = revenue_date - timedelta_object * 4
        
        # Check for revenue report days before IPO
        #if revenue_date < last_history_date:
        #    break
            
        # Search for report date in second dataframe the result returns in dataframe format
        price = 0
        df_with_price_4_days = price_df[price_df['Date']==four_days_before_revenue_date]
        df_with_price_3_days = price_df[price_df['Date']==two_days_before_revenue_date]
        df_with_price_2_days = price_df[price_df['Date']==two_days_before_revenue_date]
        df_with_price_0_days = price_df[price_df['Date']==revenue_date]
        
        # Check if dataframe is epty or not if not change price value
        # Priority to date 4 before report
        report_day = 0
        
        if len(df_with_price_0_days) > 0:
            price = df_with_price_0_days.iloc[0,4]
            report_day = 1
            
        if len(df_with_price_2_days) > 0:
            price = df_with_price_2_days.iloc[0,4]  
            report_day = 2
            
        if len(df_with_price_3_days) > 0:
            price = df_with_price_3_days.iloc[0,4] 
            report_day = 3
            
        if len(df_with_price_4_days) > 0:
            price = df_with_price_4_days.iloc[0,4]
            report_day = 4
        

        
        if report_day == 1:
            d1 = d1+1
        elif report_day == 2:
            d2 = d2+1
        elif report_day == 3:
            d3 = d3+1
        elif report_day == 4:
            d4 = d4+1
            
        
  #      Second variant with for loop (works much slower)      
  #      price = price_df[price_df['Date']==price_df.price_df[0,0]]
  #      if four_days_before_revenue_date
        
  #      for price_index, price_row in price_df.iterrows():
            # Get date from second dataframe
  #          price_date = price_df.iloc[price_index, 0]
            
  #          # If date(four days before report) found in prices DF then set 'price' var etc... 
  #          if price_date == four_days_before_revenue_date:
  #              price = price_df.iloc[price_index - 2, 4]
  #              i=1
  #          if price_date == three_days_before_revenue_date:
  #              price = price_df.iloc[price_index - 1, 4]
  #              i=2               
  #          if price_date == two_days_before_revenue_date:
  #              price = price_df.iloc[price_index, 4]
  #              i=3
  #          else:
  #              continue  
    
    
        # Built result dataframe    
        price_list.append(price)
    revenue_df['Price'] = price_list
    print("4 days enries: {}\n3 days enries: {}\n2 days enries: {}\n0 days enries: {}".format(d4, d3, d2, d1))
    print("Dataframe shape", revenue_df.shape)
    return revenue_df
```

Join dataframes using function


```python
price_rev_df = join_price_and_revenue_dfs(hist, revenue_df)
price_rev_df.head()
```

    4 days enries: 29
    3 days enries: 12
    2 days enries: 0
    0 days enries: 0
    Dataframe shape (51, 9)





<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Revenue</th>
      <th>Gross profit</th>
      <th>EPS</th>
      <th>EBITDA</th>
      <th>Operating income</th>
      <th>PE ratio</th>
      <th>Net income</th>
      <th>Price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2022-06-30</td>
      <td>28822</td>
      <td>23630</td>
      <td>2.46</td>
      <td>10337</td>
      <td>8358</td>
      <td>13.36</td>
      <td>6687</td>
      <td>160.679993</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2022-03-31</td>
      <td>27908</td>
      <td>21903</td>
      <td>2.72</td>
      <td>10680</td>
      <td>8524</td>
      <td>16.82</td>
      <td>7465</td>
      <td>229.860001</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2021-12-31</td>
      <td>33671</td>
      <td>27323</td>
      <td>3.64</td>
      <td>14599</td>
      <td>12585</td>
      <td>24.37</td>
      <td>10285</td>
      <td>346.179993</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2021-09-30</td>
      <td>29010</td>
      <td>23239</td>
      <td>3.22</td>
      <td>12418</td>
      <td>10423</td>
      <td>24.22</td>
      <td>9194</td>
      <td>340.649994</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2021-06-30</td>
      <td>29077</td>
      <td>23678</td>
      <td>3.61</td>
      <td>14353</td>
      <td>12367</td>
      <td>25.76</td>
      <td>10394</td>
      <td>355.640015</td>
    </tr>
  </tbody>
</table>
</div>



## Data preparation and exploration

First we need to cut off rows before the IPO date where 'Price' value is 0. 


```python
#Filter df using values of Price column
clean_price_rev_df = price_rev_df[price_rev_df['Price']>0]
clean_price_rev_df.tail()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Revenue</th>
      <th>Gross profit</th>
      <th>EPS</th>
      <th>EBITDA</th>
      <th>Operating income</th>
      <th>PE ratio</th>
      <th>Net income</th>
      <th>Price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>36</th>
      <td>2013-06-30</td>
      <td>1813</td>
      <td>1348</td>
      <td>0.13</td>
      <td>792</td>
      <td>562</td>
      <td>108.17</td>
      <td>331</td>
      <td>24.160000</td>
    </tr>
    <tr>
      <th>37</th>
      <td>2013-03-31</td>
      <td>1458</td>
      <td>1045</td>
      <td>0.09</td>
      <td>606</td>
      <td>373</td>
      <td>1279.00</td>
      <td>217</td>
      <td>26.090000</td>
    </tr>
    <tr>
      <th>38</th>
      <td>2012-12-31</td>
      <td>1585</td>
      <td>1187</td>
      <td>0.03</td>
      <td>747</td>
      <td>523</td>
      <td>NaN</td>
      <td>64</td>
      <td>26.049999</td>
    </tr>
    <tr>
      <th>39</th>
      <td>2012-09-30</td>
      <td>1262</td>
      <td>940</td>
      <td>-0.02</td>
      <td>553</td>
      <td>377</td>
      <td>NaN</td>
      <td>-59</td>
      <td>20.620001</td>
    </tr>
    <tr>
      <th>40</th>
      <td>2012-06-30</td>
      <td>1184</td>
      <td>817</td>
      <td>-0.08</td>
      <td>-604</td>
      <td>-743</td>
      <td>NaN</td>
      <td>-157</td>
      <td>33.099998</td>
    </tr>
  </tbody>
</table>
</div>



Then we need check for data types and set it to proper. 


```python
# Check for data types
clean_price_rev_df.dtypes
```




    Date                 object
    Revenue              object
    Gross profit         object
    EPS                  object
    EBITDA               object
    Operating income     object
    PE ratio             object
    Net income           object
    Price               float64
    dtype: object




```python
#Change Date values to Datetext first because we can't change 2000-12 text to numeric format
clean_price_rev_df[['Date']] = clean_price_rev_df[['Date']].apply(pd.to_datetime)

#Change data types to Numeric
clean_price_rev_df = clean_price_rev_df.apply(pd.to_numeric)

#Change Date values from Numeric to Datetext again
clean_price_rev_df[['Date']] = clean_price_rev_df[['Date']].apply(pd.to_datetime)
```

    /home/timur/anaconda3/lib/python3.9/site-packages/pandas/core/frame.py:3069: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      self[k1] = value[k2]



```python
# Check for data types
clean_price_rev_df.dtypes
```




    Date                datetime64[ns]
    Revenue                    float64
    Gross profit               float64
    EPS                        float64
    EBITDA                     float64
    Operating income           float64
    PE ratio                   float64
    Net income                 float64
    Price                      float64
    dtype: object




```python
# Check for NaN values
clean_price_rev_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 41 entries, 0 to 40
    Data columns (total 9 columns):
     #   Column            Non-Null Count  Dtype         
    ---  ------            --------------  -----         
     0   Date              41 non-null     datetime64[ns]
     1   Revenue           41 non-null     float64       
     2   Gross profit      41 non-null     float64       
     3   EPS               41 non-null     float64       
     4   EBITDA            41 non-null     float64       
     5   Operating income  41 non-null     float64       
     6   PE ratio          38 non-null     float64       
     7   Net income        41 non-null     float64       
     8   Price             41 non-null     float64       
    dtypes: datetime64[ns](1), float64(8)
    memory usage: 3.2 KB


As we can see PE ration has 3 NaN values lets replace it by mean value


```python
# Replace NaN by Mean
clean_price_rev_df.fillna(value=clean_price_rev_df['PE ratio'].mean(), inplace=True)
```

The theory is that after the report date we should wait aproximately 3 month to see  how this report affected to the market price. So current report data will be used for predictions to the next 3 month.
For this purposes we need shif price column down to 1 row.


```python
# Clone dataframe
price_rev_df_shifted = clean_price_rev_df 
```


```python
# Shift price column
price_rev_df_shifted['Price'] = price_rev_df_shifted['Price'].shift(1)
price_rev_df_shifted.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Revenue</th>
      <th>Gross profit</th>
      <th>EPS</th>
      <th>EBITDA</th>
      <th>Operating income</th>
      <th>PE ratio</th>
      <th>Net income</th>
      <th>Price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2022-06-30</td>
      <td>28822.0</td>
      <td>23630.0</td>
      <td>2.46</td>
      <td>10337.0</td>
      <td>8358.0</td>
      <td>13.36</td>
      <td>6687.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2022-03-31</td>
      <td>27908.0</td>
      <td>21903.0</td>
      <td>2.72</td>
      <td>10680.0</td>
      <td>8524.0</td>
      <td>16.82</td>
      <td>7465.0</td>
      <td>160.679993</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2021-12-31</td>
      <td>33671.0</td>
      <td>27323.0</td>
      <td>3.64</td>
      <td>14599.0</td>
      <td>12585.0</td>
      <td>24.37</td>
      <td>10285.0</td>
      <td>229.860001</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2021-09-30</td>
      <td>29010.0</td>
      <td>23239.0</td>
      <td>3.22</td>
      <td>12418.0</td>
      <td>10423.0</td>
      <td>24.22</td>
      <td>9194.0</td>
      <td>346.179993</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2021-06-30</td>
      <td>29077.0</td>
      <td>23678.0</td>
      <td>3.61</td>
      <td>14353.0</td>
      <td>12367.0</td>
      <td>25.76</td>
      <td>10394.0</td>
      <td>340.649994</td>
    </tr>
  </tbody>
</table>
</div>



Now it's time to explore our dataframe. Let's take a look to correlation between variables.


```python
# Build correlation matrix
clean_price_rev_df.corr()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Revenue</th>
      <th>Gross profit</th>
      <th>EPS</th>
      <th>EBITDA</th>
      <th>Operating income</th>
      <th>PE ratio</th>
      <th>Net income</th>
      <th>Price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Revenue</th>
      <td>1.000000</td>
      <td>0.999351</td>
      <td>0.957176</td>
      <td>0.972385</td>
      <td>0.958794</td>
      <td>-0.292468</td>
      <td>0.950839</td>
      <td>0.889512</td>
    </tr>
    <tr>
      <th>Gross profit</th>
      <td>0.999351</td>
      <td>1.000000</td>
      <td>0.960797</td>
      <td>0.976668</td>
      <td>0.964345</td>
      <td>-0.299390</td>
      <td>0.955033</td>
      <td>0.892772</td>
    </tr>
    <tr>
      <th>EPS</th>
      <td>0.957176</td>
      <td>0.960797</td>
      <td>1.000000</td>
      <td>0.990086</td>
      <td>0.990158</td>
      <td>-0.286533</td>
      <td>0.999506</td>
      <td>0.894400</td>
    </tr>
    <tr>
      <th>EBITDA</th>
      <td>0.972385</td>
      <td>0.976668</td>
      <td>0.990086</td>
      <td>1.000000</td>
      <td>0.998256</td>
      <td>-0.299186</td>
      <td>0.989004</td>
      <td>0.902068</td>
    </tr>
    <tr>
      <th>Operating income</th>
      <td>0.958794</td>
      <td>0.964345</td>
      <td>0.990158</td>
      <td>0.998256</td>
      <td>1.000000</td>
      <td>-0.296294</td>
      <td>0.989972</td>
      <td>0.892217</td>
    </tr>
    <tr>
      <th>PE ratio</th>
      <td>-0.292468</td>
      <td>-0.299390</td>
      <td>-0.286533</td>
      <td>-0.299186</td>
      <td>-0.296294</td>
      <td>1.000000</td>
      <td>-0.288899</td>
      <td>-0.335888</td>
    </tr>
    <tr>
      <th>Net income</th>
      <td>0.950839</td>
      <td>0.955033</td>
      <td>0.999506</td>
      <td>0.989004</td>
      <td>0.989972</td>
      <td>-0.288899</td>
      <td>1.000000</td>
      <td>0.897509</td>
    </tr>
    <tr>
      <th>Price</th>
      <td>0.889512</td>
      <td>0.892772</td>
      <td>0.894400</td>
      <td>0.902068</td>
      <td>0.892217</td>
      <td>-0.335888</td>
      <td>0.897509</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



What do we realise here? There is strong correlation between varibles. I'm not surprized because that vars shows almoust same measure Income of the company. Strong correlation I mean value of correlation coefficien more than 90%. To automate process of determining vars that have strong correlation lets use function:


```python
def get_vars_list_with_strong_correlation(dataframe):
    # Function returns list of vars that correlates to each  other with value of correlation
    # coefficien more then 90%
    
    # Create an empty list
    CorField = []
    
    # Count correlation coefficients
    CorrKoef=dataframe.corr()    
    
    # For each column in df
    for column_index in CorrKoef:
        
        # For each var index in filtered df where value more then 90%
        for var_index in CorrKoef.index[CorrKoef[column_index] > 0.9]:
            
            # Check if var index already in list and check for coefficient 
            # for same variable
            if column_index != var_index and var_index not in CorField and column_index not in CorField:
                CorField.append(var_index)
                print ("%s-->%s: r^2=%f" % (column_index,var_index, CorrKoef[column_index][CorrKoef.index==var_index].values[0]))
    return(CorField)           
```


```python
# Get vars that has strong correlation
xx = get_vars_list_with_strong_correlation(clean_price_rev_df)
xx
```

    Revenue-->Gross profit: r^2=0.999351
    Revenue-->EPS: r^2=0.957176
    Revenue-->EBITDA: r^2=0.972385
    Revenue-->Operating income: r^2=0.958794
    Revenue-->Net income: r^2=0.950839





    ['Gross profit', 'EPS', 'EBITDA', 'Operating income', 'Net income']



Now wee need to exclude 5 of this 6 vars. Let's take a look to correlation of this vars to 'Price'. As we can see strongest correlation with EBITDA. So we'll use EBITDA and PE ratio to our model.

## Linear model prediction

Since we have dataframe and know that there is linear correlation between <b>price</b> and <b>revenue</b> we can build linear regression model. Let's jump right in.


```python
# Import libraries
import numpy as np
# Preprocessing allows us to standarsize our data
from sklearn import preprocessing
# Allows us to split our data into training and testing data
from sklearn.model_selection import train_test_split
# Linear model algorithm
from sklearn import linear_model
```

Cut of first row with NaN and date column.


```python
price_rev_df_shifted_without_first_row = price_rev_df_shifted[1:len(price_rev_df_shifted)]
price_rev_df_shifted_without_first_row.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Revenue</th>
      <th>Gross profit</th>
      <th>EPS</th>
      <th>EBITDA</th>
      <th>Operating income</th>
      <th>PE ratio</th>
      <th>Net income</th>
      <th>Price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>2022-03-31</td>
      <td>27908.0</td>
      <td>21903.0</td>
      <td>2.72</td>
      <td>10680.0</td>
      <td>8524.0</td>
      <td>16.82</td>
      <td>7465.0</td>
      <td>160.679993</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2021-12-31</td>
      <td>33671.0</td>
      <td>27323.0</td>
      <td>3.64</td>
      <td>14599.0</td>
      <td>12585.0</td>
      <td>24.37</td>
      <td>10285.0</td>
      <td>229.860001</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2021-09-30</td>
      <td>29010.0</td>
      <td>23239.0</td>
      <td>3.22</td>
      <td>12418.0</td>
      <td>10423.0</td>
      <td>24.22</td>
      <td>9194.0</td>
      <td>346.179993</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2021-06-30</td>
      <td>29077.0</td>
      <td>23678.0</td>
      <td>3.61</td>
      <td>14353.0</td>
      <td>12367.0</td>
      <td>25.76</td>
      <td>10394.0</td>
      <td>340.649994</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2021-03-31</td>
      <td>26171.0</td>
      <td>21040.0</td>
      <td>3.30</td>
      <td>13350.0</td>
      <td>11378.0</td>
      <td>25.20</td>
      <td>9497.0</td>
      <td>355.640015</td>
    </tr>
  </tbody>
</table>
</div>




```python
price_rev_first_row = price_rev_df_shifted[0:1]
price_rev_first_row
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Revenue</th>
      <th>Gross profit</th>
      <th>EPS</th>
      <th>EBITDA</th>
      <th>Operating income</th>
      <th>PE ratio</th>
      <th>Net income</th>
      <th>Price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2022-06-30</td>
      <td>28822.0</td>
      <td>23630.0</td>
      <td>2.46</td>
      <td>10337.0</td>
      <td>8358.0</td>
      <td>13.36</td>
      <td>6687.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



Create train and test dataset


```python
# Split test and train data
X = price_rev_df_shifted_without_first_row[['EBITDA', 'PE ratio']].values
Y = price_rev_df_shifted_without_first_row['Price'].values
                                           
from sklearn.model_selection import train_test_split 

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)
```

    Train set: (32, 2) (32,)
    Test set: (8, 2) (8,)


When train and test data splited build the model


```python
# Built the model
regr = linear_model.LinearRegression()

regr.fit (X_train, y_train)

# The coefficients
print ('Coefficients: ', regr.coef_)
print('Intercept: ',regr.intercept_) 
```

    Coefficients:  [ 0.01709512 -0.03437189]
    Intercept:  54.283655339925076



```python
# Predict values
y_test_predicted = regr.predict(X_test)

# Built df with actual and predicted values
actual_predicted = pd.DataFrame({'Actual': y_test, 'Predicted': y_test_predicted}) 
print(actual_predicted)

print("\nResidual sum of squares: %.2f"
      % np.mean((y_test_predicted - y_test) ** 2))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(X_test, y_test))
```

           Actual   Predicted
    0  118.010002  115.969915
    1  153.589996  121.338244
    2  116.139999  104.519219
    3  355.640015  281.637316
    4  208.100006  200.340890
    5  166.949997  171.154781
    6   88.010002   75.302306
    7  180.110001  157.918237
    
    Residual sum of squares: 923.45
    Variance score: 0.85



```python
from sklearn import metrics 

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_test_predicted)) 
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_test_predicted)) 
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_test_predicted)))
```

    Mean Absolute Error: 20.84733503146682
    Mean Squared Error: 923.4529342108942
    Root Mean Squared Error: 30.38836840323768



```python
y_hat= regr.predict(price_rev_first_row[['EBITDA', 'PE ratio']])
```


```python
print("Predicted price: ",y_hat, "Current price: ", hist.iloc[len(hist)-1, 4])
```

    Predicted price:  [230.53668715] Current price:  158.5399932861328


## Conclusion

Now when we know predicted price we can use that information to make investment desigions. But this information only gives general view to fair price of the stock.

Created by Timur Talikbayev
