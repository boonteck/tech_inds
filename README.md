# tech_inds
Build indicators for stock markets for technical analysis. Optimized with Numba, takes and returns numpy arrays as outputs.

This is an experimental piece aimed at understanding usage of Numba and calculation of common technical indicators.
For a comprehensive package with faster implementation in Cython, see https://github.com/mrjbq7/ta-lib

<h3>Financial Indicators included:</h3>

1. Simple Moving Average                      _*SMA*_
2. Exponential Moving Average                 _*EMA*_
3. Moving Standard Deviation                  _*stdev*_
4. Price Rate of Change                       _*ROC*_
5. Bollinger Bands                            _*Boll*_
6. Average True Range                         _*ATR*_
7. Moving Average Convergence/Divergence      _*MACD*_
8. Relative Strength Index                    _*RSI*_
9. Commodity Channel Index                    _*CCI*_
10. Stochastic Momentum Index                 _*SMI*_

### Install 
```
sudo pip install git+https://github.com/boonteck/tech_inds.git
```
