import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor

monthlist = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
             'August', 'September', 'October', 'November', 'December']

weekdaylist = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday',
               'Saturday', 'Sunday']

orangeline = ['Oak Grove', 'Malden Center', 'Wellington', 'Assembly',
              'Sullivan Square', 'Community College', 'North Station',
              'Haymarket', 'State', 'Downtown Crossing', 'Chinatown',
              'Tufts Medical Center', 'Back Bay', 'Massachusetts Avenue',
              'Ruggles', 'Roxbury Crossing', 'Jackson Square', 'Stony Brook',
              'Green Street', 'Forest Hills']

orangealph = ['Assembly', 'Back Bay', 'Chinatown', 'Community College', 
              'Downtown Crossing', 'Forest Hills', 'Green Street', 'Haymarket',
              'Jackson Square', 'Malden Center', 'Massachusetts Avenue',
              'North Station', 'Oak Grove', 'Roxbury Crossing', 'Ruggles',
              'State', 'Stony Brook', 'Sullivan Square', 'Tufts Medical Center',
              'Wellington']

def read_and_setup(filename):
    # Gets the intial file setup
    table = pd.read_csv(filename)
    table = table.dropna()
    table = pd.get_dummies(table, columns=['stop_name'])
    return table


def get_stops(start, end, direction):
    # Makes a list of all the stops that the train will be passing from the
    # first stop, last stop, and 
    stopslist = []
    firstindex = orangeline.index(start)
    lastindex = orangeline.index(end)
    if direction == 0:
        change = 1
    else:
        change = -1
    count = firstindex
    while count != lastindex:
        stopslist.append(orangeline[count])
        count += change
    return stopslist

def get_user_inputs():
    
    # Asks for user inputs and initializes the variables for the user predicitons
    leaving_station = 0
    arriving_station = 0
    
    # Getting departure station
    while leaving_station not in orangeline:
        leaving_station = input('What station are you departing? ')
        if leaving_station.upper() == 'HELP':
            print(orangeline)
        elif leaving_station not in orangeline:
            print('Please enter a valid station. Make sure to enter the', 
                  'full station name (for example, Mass Ave should be', 
                  'Massachusetts Avenue) If you\'d like to see the full',
                  'list of valid stops, please enter "HELP".')
   
    # Geting arriving station
    while arriving_station not in orangeline:
        arriving_station = input('What station are you going to? ')
        if arriving_station.upper() == 'HELP':
            print(orangeline)
        elif arriving_station not in orangeline:
            print('Please enter a valid station. Make sure to enter the', 
                  'full station name (for example, Mass Ave should be', 
                  'Massachusetts Avenue) If you\'d like to see the full',
                  'list of valid stops, please enter "HELP".')
        elif arriving_station == leaving_station:
            print('Please enter a station that is different than the one,'
                  'you are departing.')
            arriving_station = 0
     
    # Getting the day of the week
    weekday = 0
    while weekday not in weekdaylist:
        weekday = input('What day of the week is your train on? ')
        if weekday not in weekdaylist:
            print('Please enter a valid day.')
    weekday = weekdaylist.index(weekday)
    
    # Getting the month
    month = 0
    while month not in monthlist:
        month = input('What month is your train ride in? ')
        if month not in monthlist:
            print('Please enter a valid month.')
    month = monthlist.index(month) + 1
    
    # Getting the temperature
    while True:
        temp = input('What temperature do you expect it to be?' +
                     ' Please enter a number in degrees Fahrenheit. ' )
        try:
            int(temp)
        except:
            print('Please enter a valid integer')
        else:
            temp = int(temp)
            break
    
    # Getting the precipitation
    while True:
        precip = input('Roughly how many inches of precipitation do ' +
                       'you expect? Please enter 0 if none or unsure. ')
        try:
            float(precip)
        except:
            print('Please enter a valid integer')
        else:
            precip = float(precip)
            break
    
    # Getting the direction
    if orangeline.index(leaving_station) < orangeline.index(arriving_station):
        direction = 0
    else:
        direction = 1
        
    stops_list = get_stops(leaving_station, arriving_station, direction)
    
    X_list = []
    
    # Opening the comparison file so we can get medians
    comparison = pd.read_csv('OrangeLineAverages.csv')
    comparison = comparison[comparison['direction_id'] == direction]
    comparisonlist = []
    
    for stop in stops_list:
        
        # Grabbing median for the stop and putting that in its own list
        temptable = comparison[comparison['stop_name'] == stop]
        comparisonlist.append(temptable.iloc[0][3])
        
        # Creating an array that can be used with the algorithms
        reglist = [direction, temp, month, precip, weekday]
        stop = orangealph.index(stop)
        indexlist = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        indexlist[stop] = 1
        for item in indexlist:
            reglist.append(item)
            
        # Appending the items to the list in a way that they can be interpreted
        # by the ML models
        X_list.append(reglist)
        
    return X_list, comparisonlist


def initializeregression(table):
    # Initializes the knn regression
    X   = table[['direction_id', 'Temp_Avg', 'month', 'Precipitation', 
                 'Day_of_Week', 'stop_name_Assembly', 'stop_name_Back Bay', 
                 'stop_name_Chinatown', 'stop_name_Community College',
                 'stop_name_Downtown Crossing', 'stop_name_Forest Hills', 
                 'stop_name_Green Street', 'stop_name_Haymarket', 
                 'stop_name_Jackson Square', 'stop_name_Malden Center', 
                 'stop_name_Massachusetts Avenue', 'stop_name_North Station',      
                 'stop_name_Oak Grove', 'stop_name_Roxbury Crossing', 
                 'stop_name_Ruggles', 'stop_name_State', 
                 'stop_name_Stony Brook', 'stop_name_Sullivan Square', 
                 'stop_name_Tufts Medical Center', 'stop_name_Wellington']]
    y   = table['slow_day_flag']
    knn = KNeighborsClassifier(n_neighbors = 3)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size= 0.2,
                                                    random_state= 4)
    knn.fit(X, y)
    return knn


def initializerf(table):
    # Initializes the random forest regression
    X   = table[['direction_id', 'Temp_Avg', 'month', 'Precipitation', 'Day_of_Week', 'stop_name_Assembly', 'stop_name_Back Bay', 'stop_name_Chinatown', 'stop_name_Community College',
             'stop_name_Downtown Crossing', 'stop_name_Forest Hills', 'stop_name_Green Street', 'stop_name_Haymarket', 'stop_name_Jackson Square', 
             'stop_name_Malden Center', 'stop_name_Massachusetts Avenue', 'stop_name_North Station', 'stop_name_Oak Grove', 'stop_name_Roxbury Crossing', 
             'stop_name_Ruggles', 'stop_name_State', 'stop_name_Stony Brook', 'stop_name_Sullivan Square', 'stop_name_Tufts Medical Center', 'stop_name_Wellington']]
    y   = table['delay_time']
    rf = RandomForestRegressor(n_estimators = 100, random_state=7)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state= 4)
    rf.fit(X_train, y_train)
    return rf

def regression(knn, wanted_value):
    pred = knn.predict([wanted_value])
    return pred[0]


def rfregression(rf, wanted_value):
    pred = rf.predict([wanted_value])
    return pred[0]
    
def evaluateresults(predictionlist, predictionlist_knn, comparisonlist):
    # Evaluating the results
    
    # Printing the likelihood of a delay
    total = len(predictionlist_knn)
    slows = predictionlist_knn.count(1)
    if slows / total > .75:
        print('\nLeave early- there\'s a high chance of delays.')
    elif slows / total < .3:
        print('\nDelays are very unlikely.')
    else:
        print('\nThere\'s a small chance of delays, but you should be fine.')

    # Printing the expected travel time
    mediansum = sum(comparisonlist)
    predictedsum = sum(predictionlist)
    difference = predictedsum - mediansum
    print('Normally, your trip would take', round(mediansum / 60, 2), 'minutes.')
    if difference >= 0:
        print('Regardless of the likelihood of a delay, we expect your train',
              'to run', round(difference / 60, 2), 'minutes slower than normal.')

    else:
        print('Regardless of the likelihood of a delay, we expect your train', 
              'to run', abs(round(difference/60, 2)), 'minutes faster than normal.')
    print('We expect the trip to take', round(predictedsum/60, 2), 'minutes.')
if __name__ == "__main__":
    table = read_and_setup('OrangeLineDailyAverages.csv')
    knn = initializeregression(table)
    rf = initializerf(table)
    testlist, comparisonlist = get_user_inputs()
    predictionlist = []
    predictionlist_knn = []
    for wanted_value in testlist:
        predictionlist_knn.append(regression(knn, wanted_value))
        predictionlist.append(rfregression(rf, wanted_value))
    evaluateresults(predictionlist, predictionlist_knn, comparisonlist)
