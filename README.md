# Predicting effective temperature in a chicken barn 
## Part II Benchmarking timeseries prediction against non-linear regression

Meat chickens, also called broilers, in intensive rearing live in closed, controlled environment for about 36-42 days (depending on the target market) before they are brought to the abattoir. During this time, the thermal environment is very important, as chickens have no sweat glands and are prone to suffer from both cold and hot temperatures. Chickens manage their core temperature through heat exchange via their feet and their respiration, but are able to do so only after about 10 days of life. The so-called *thermo-neutral* zone of a chicken, i.e. the thermal range that allows a chicken to maintain a constant core temperature grows over time - the exact ranges are not currently known.

Chickens like humans are affected not only by temperature: humidity also plays a great role. The higher the humidity, the more extreme temperature is perceived, because the higher water contents in the air conducts the heat better. Air flow or air speed also pays an important role: the higher the air speed, the colder it feels. This is called the *windchill effect*. For chickens, our best measure currently is called the *effective temperature* and is a combination of temperature, relative humidity and airspeed.

In order to cater for the different needs of birds during their lifetime, farmers have targets for temperature, relative humidity and some also for airspeed.

Modern barns operate heating and cooling elements automatically via sensor input. The so-called "controller", essentially a computer, uses inputs from sensors and translates them into actions: starting/stopping the heater, ventilators, air flaps and cooling elements. In essence, the controller compares the sensor input to the targets and decides on the action to be taken based on that information. Given that farms are quite large spaces (2,500m<sup>2</sup> x 3m height = 7,500m<sup>3</sup> is not atypical), controllers try to anticipate changes in the ambient conditions to avoid reaching suboptimal conditions for the animals.

In cold weather, farms try to ventilate less - because then they have to heat the entering cold air which increases the cost of production. In hot weather, farmers use laminar ventilation (called "tunnel ventilation") to employ the windchill effect in cooling down the birds. Farms in hot climate also operate heat exchangers or coolers to mitigate outside conditions.

If controllers were perfect and barns were perfect, the ambient conditions inside would be indentical with the targets. However, that is not true. The sun, for example, has significant impact by heating up the barn; this triggers higher ventilation rates and lower temperatures (or effective temperatures) in the afternoons. At the same time, while heating does change the relative humidity (it dries the air), heating is expensive. If the outside relative humidity is too high, but the temperature is acceptable, controllers will allow for a higher relative humidity. As controller typically do not operate on effective temperature, this means that the thermal sensation will be stronger (colder or hotter).

Controllers heat and cool the environment based on sensor input. However, the change of the temperature sensation in the barn is slow, so predicting the conditions some hours in advance will allow controllers to slowly ramp up their heating and cooling and therefore save energy while maintaining the barn always at optimum conditions.

This is **Part 2** of the analysis of the thermal sensation in a chicken barn. In Part 1, a correlation analysis and linear regression was undertaken to understand the impact of outside conditions on the thermal sensation as measured by the *effective temperature*. 

The goal of this notebook is to:
1) Analyse the problem from a timeseries perspective
2) Build a predictor of the effective temperature that is capable of predicting the effective temperature inside the barn 4-12h in advance with an error <2ºC

Three data sets are available (two for training and one for testing).

## Is this a timeseries problem or a (non-linear) regression problem?

The question says it all: in principle, chickens are raised in (discontinuous) flocks. In the absence of outside influence, flocks are *meant* to be identical, i.e. every day of production should find the same stage of development and therefore the same thermal needs of the birds. Therefore the thermal conditions should be the same. We can allow for some outside influence by weather, but that should really explain all the differences. This points toward a regression problem.

On the other hand, there is a variety of variables that exist but we have no access to. Farmers might change the settings of their climate system, the response of such system might not be linear and there might be weather factors (such as hours and intensity of sunlight) that we have no access to. Also, we have no knowledge how the farmers adapts to changes in genetics of the birds and/or advice by external consultants, vets etc. In addition, it is very clear that conditions in the farm are autoregressive. None of the conditions change very fast, so clearly e.g. the temperature at time <em>t</em> is correlated with the temperature at times <em>t-n</em>. For these reasons, we could look at this as a (discontinuous) timeseries problem.

This analysis benchmarks both approaches. For the timeseries approach we will tack on one timeseries to another, thereby creating an artificial, continuous timeseries. For the prediction, we will have to translate timestamps back to the original time. The timeseries is modeled using AutoGluon (https://auto.gluon.ai/).

For the regression approach we will include the autoregression by adding past values as features into the data set. Regression is executed with XGBoost and LightGBM. LightGBM performs better on this task than XGBoost.

# Analysis of results and summary

<table>
    <tr>
        <td><img src="https://cdn-icons-png.flaticon.com/512/9746/9746435.png" alt="Success!" width="70%" height="auto"></td>
        <td>
        <h3><b>Mission accomplished - twice!</b></h3>
        Predicting an uncertain future is more complicated than interpolating using a (supposedly) complete model of the world. It is hardly surprising that the timeseries approach was not as successful as the regression approach - and what this really tells us is that the chicken barn is well approximated as a closed system. Surely, adding other, relevant parameters like the setpoint temperature or the heating/cooling activity would make our model a lot better, but as it stands, we are able to predict the inside temperature of the barn 12h in advance with only the weather variables at time <em>t ... t-5</em>. The root mean square error of the 12h prediction is about 1ºC [&cong; 2 ºF] which is smaller than the variation during a day and in different parts of the barn. The timeseries forecast had a root mean square error of 1.76, so was significantly higher. However, it both cases the target set in the problem statement (error < 2ºC) was met.</td>
    </tr>
    <tr>
        <td><img src="https://cdn3.iconfinder.com/data/icons/business-management-part-3-2/512/40-1024.png" alt="Next steps" width="70%" height="auto"></td>
        <td>
        <h3><b>Next steps</b></h3>
        The analysis would need to be redone on different prediction "slices"; it might also help decrease the error of the timeseries prediction if we predicted only a single period and then re-trained the model with the prediction until we reach 12h. However, computationally this would be quite demanding as AutoGluon is not fast (for obvious reasons) and it is not totally clear whether this is not already done internally by AutoGluon - the documentation is still catching up a bit.

After that, the next logical step would be to extend the training set to at least 2 flocks per season and then see how the prediction accuracy changes. </td>
    </tr>
    <tr>
        <td><img src="https://cdn-icons-png.flaticon.com/512/3930/3930474.png" alt="Improvements" width="70%" height="auto"></td>
        <td>
        <h3><b>Improvements</b></h3>
        There are many ways to improve this analysis, probably in both "branches". For the timeseries analysis, we have used AutoGluon out of the box without trying to optimise any of the underlying models. AutoGluon provides us with an ensemble model with consistently the best results, but it is not clear how to obtain the same model "manually". For the regression problem, a similiar statement is true - we have not optimised the model to the fullest (and only tried LightGBM and XGBoost).</td>
    </tr>
</table>

