require(fpp)

# Loading the Data
data(elecequip)
plot(elecequip)

# Get the STL plot to analyse trend, seasonality and noise
newElecEquip <- stl(elecequip,s.window = "periodic")
plot(newElecEquip)
# data is seasonal

# Season Adjust
elecequip.sa = seasadj(newElecEquip)
plot(stl(elecequip.sa,s.window = 12))

# Plotting the season adjusted data
plot(elecequip.sa)
plot(stl(elecequip.sa,s.window = 12))
#there is no need to stabilize the variance as the variance is alreay stable.

# Check if the data is Stationery using ACF plot and Box.test
acf(elecequip.sa)
Box.test(elecequip.sa)
plot(elecequip.sa)
#Since the acf of the data is decreasing slowly, the data is non stationery.
#Also the plot is not horizontal and contains trend and seasonality so non stationery.

# Apply differencing to convert non stationery data to stationery
diffelecequip.sa <- diff(elecequip.sa)
acf(diffelecequip.sa)
#the differencing converted non stationery data to stationery.

# Fit the model using auto.arima() on stationery data
fit <- auto.arima(diffelecequip.sa)
summary(fit)
# p=3, d=0, q=1

# Check if any other arima model works better
arima1 = Arima(diffelecequip.sa,order = c(4,0,0))
arima2 = Arima(diffelecequip.sa,order = c(3,0,0))
arima3 = Arima(diffelecequip.sa,order = c(2,0,0))
arima4 = Arima(diffelecequip.sa,order = c(3,0,1))
# arima4 gives better AICc, which is the model we got from auto.arima()

# Check for residuals
Acf(residuals(arima4))
Box.test(residuals(arima4))
#checked if residual is white noise, the model is ready for forecast.

# Plotting the forecast
plot(forecast(arima4))