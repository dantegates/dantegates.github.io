---
layout: post
mathjax: true
title: MLB 2020 Postseason Projections
github: https://github.com/dantegates/mlb-statcast
creation_date: 2020-09-29
last_modified: 2020-09-29 13:47:13
tags: 
  - MLB
  - Bayesian Inference
---


Just over 6 months after the 2020 MLB season was postponed indefinitely and just under 3 months after the 60-game schedule was announced the 2020 postseason begins today. While MLB postseason results are often compared to a crapshoot, it doesn't stop us from trying.

In 2018 [I posted projections](https://dantegates.github.io/2018/10/22/world-series-projections.html) for the Dodgers-Red Sox series based on [an even earlier post](https://dantegates.github.io/2018/09/20/hierarchical-bayesian-ranking.html) on probabilistic ranking.

This year, I've taken the 2018 model (with a few small tweaks) and used it to simulate the entire 2020 postseason. The table below shows the results, with the Dodgers and Rays, unsurprisingly, topping the  list as World Series  favorites. Note that teams that did *not* make the postseason are also  included, with a 0% chance of any success.





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th></th>
      <th colspan="4" halign="left">Probability of Becomming</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th>Wild Card Champion</th>
      <th>Division Champion</th>
      <th>League Champion</th>
      <th>World Series Champion</th>
    </tr>
    <tr>
      <th>Team</th>
      <th>Division</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>LAD</th>
      <th>NL West</th>
      <td>74.68</td>
      <td>50.98</td>
      <td>38.32</td>
      <td>26.24</td>
    </tr>
    <tr>
      <th>TB</th>
      <th>AL East</th>
      <td>64.00</td>
      <td>39.22</td>
      <td>24.91</td>
      <td>13.93</td>
    </tr>
    <tr>
      <th>SD</th>
      <th>NL West</th>
      <td>61.98</td>
      <td>27.38</td>
      <td>17.13</td>
      <td>9.39</td>
    </tr>
    <tr>
      <th>MIN</th>
      <th>AL Central</th>
      <td>64.59</td>
      <td>33.90</td>
      <td>16.55</td>
      <td>7.76</td>
    </tr>
    <tr>
      <th>CWS</th>
      <th>AL Central</th>
      <td>51.91</td>
      <td>28.39</td>
      <td>13.89</td>
      <td>6.36</td>
    </tr>
    <tr>
      <th>CLE</th>
      <th>AL Central</th>
      <td>53.11</td>
      <td>24.98</td>
      <td>12.74</td>
      <td>5.69</td>
    </tr>
    <tr>
      <th>ATL</th>
      <th>NL East</th>
      <td>56.37</td>
      <td>30.68</td>
      <td>11.70</td>
      <td>5.40</td>
    </tr>
    <tr>
      <th>OAK</th>
      <th>AL West</th>
      <td>48.09</td>
      <td>25.54</td>
      <td>11.82</td>
      <td>4.84</td>
    </tr>
    <tr>
      <th>CHC</th>
      <th>NL Central</th>
      <td>53.79</td>
      <td>27.78</td>
      <td>10.28</td>
      <td>4.48</td>
    </tr>
    <tr>
      <th>NYY</th>
      <th>AL East</th>
      <td>46.89</td>
      <td>19.50</td>
      <td>9.11</td>
      <td>3.71</td>
    </tr>
    <tr>
      <th>MIA</th>
      <th>NL East</th>
      <td>46.21</td>
      <td>20.99</td>
      <td>6.92</td>
      <td>2.63</td>
    </tr>
    <tr>
      <th>TOR</th>
      <th>AL East</th>
      <td>36.00</td>
      <td>16.30</td>
      <td>7.22</td>
      <td>2.58</td>
    </tr>
    <tr>
      <th>CIN</th>
      <th>NL Central</th>
      <td>43.63</td>
      <td>20.55</td>
      <td>6.67</td>
      <td>2.50</td>
    </tr>
    <tr>
      <th>STL</th>
      <th>NL Central</th>
      <td>38.02</td>
      <td>12.26</td>
      <td>5.44</td>
      <td>2.06</td>
    </tr>
    <tr>
      <th>HOU</th>
      <th>AL West</th>
      <td>35.41</td>
      <td>12.17</td>
      <td>3.76</td>
      <td>1.23</td>
    </tr>
    <tr>
      <th>MIL</th>
      <th>NL Central</th>
      <td>25.32</td>
      <td>9.38</td>
      <td>3.54</td>
      <td>1.20</td>
    </tr>
    <tr>
      <th>SEA</th>
      <th>AL West</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>SF</th>
      <th>NL West</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>PIT</th>
      <th>NL Central</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>TEX</th>
      <th>AL West</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>ARI</th>
      <th>NL West</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>PHI</th>
      <th>NL East</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>NYM</th>
      <th>NL East</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>LAA</th>
      <th>AL West</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>KC</th>
      <th>AL Central</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>DET</th>
      <th>AL Central</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>COL</th>
      <th>NL West</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>BOS</th>
      <th>AL East</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>BAL</th>
      <th>AL East</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>WSH</th>
      <th>NL East</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
  </tbody>
</table>
</div>



# Appendix

## Other  projections


For further context, you can compare my projections with [mlb.com's](https://www.mlb.com/news/2020-mlb-postseason-predictions) projections as well as [fivethirtyeight's](https://projects.fivethirtyeight.com/2020-mlb-predictions/). To convert mlb.com's expert predictions to probabilities I counted each time an  analyst projected  a team to  win a title and divided by the number of analysts (12).





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th></th>
      <th colspan="4" halign="left">DG: Probability of Becomming</th>
      <th colspan="4" halign="left">MLB: Probability of Becomming</th>
      <th colspan="4" halign="left">fivethirtyeight: Probability of Becomming</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th>Wild Card Champion</th>
      <th>Division Champion</th>
      <th>League Champion</th>
      <th>World Series Champion</th>
      <th>Wild Card Champion</th>
      <th>Division Champion</th>
      <th>League Champion</th>
      <th>World Series Champion</th>
      <th>Wild Card Champion</th>
      <th>Division Champion</th>
      <th>League Champion</th>
      <th>World Series Champion</th>
    </tr>
    <tr>
      <th>Team</th>
      <th>Division</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
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
      <th>LAD</th>
      <th>NL West</th>
      <td>73.86</td>
      <td>49.43</td>
      <td>36.67</td>
      <td>24.65</td>
      <td>100.0</td>
      <td>92.0</td>
      <td>92.0</td>
      <td>75.0</td>
      <td>77.0</td>
      <td>58.0</td>
      <td>45.0</td>
      <td>32.0</td>
    </tr>
    <tr>
      <th>TB</th>
      <th>AL East</th>
      <td>63.30</td>
      <td>38.90</td>
      <td>24.44</td>
      <td>13.66</td>
      <td>92.0</td>
      <td>50.0</td>
      <td>58.0</td>
      <td>17.0</td>
      <td>69.0</td>
      <td>38.0</td>
      <td>21.0</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>SD</th>
      <th>NL West</th>
      <td>61.89</td>
      <td>28.45</td>
      <td>17.63</td>
      <td>9.56</td>
      <td>83.0</td>
      <td>8.0</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>59.0</td>
      <td>20.0</td>
      <td>11.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>MIN</th>
      <th>AL Central</th>
      <td>63.83</td>
      <td>33.40</td>
      <td>16.99</td>
      <td>7.92</td>
      <td>100.0</td>
      <td>75.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>55.0</td>
      <td>32.0</td>
      <td>17.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>CWS</th>
      <th>AL Central</th>
      <td>51.68</td>
      <td>27.90</td>
      <td>13.17</td>
      <td>6.00</td>
      <td>58.0</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>43.0</td>
      <td>17.0</td>
      <td>7.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>ATL</th>
      <th>NL East</th>
      <td>56.78</td>
      <td>31.64</td>
      <td>12.20</td>
      <td>5.97</td>
      <td>25.0</td>
      <td>17.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>58.0</td>
      <td>35.0</td>
      <td>13.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>CLE</th>
      <th>AL Central</th>
      <td>53.07</td>
      <td>24.62</td>
      <td>12.32</td>
      <td>5.42</td>
      <td>25.0</td>
      <td>8.0</td>
      <td>8.0</td>
      <td>8.0</td>
      <td>46.0</td>
      <td>21.0</td>
      <td>10.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>OAK</th>
      <th>AL West</th>
      <td>48.32</td>
      <td>25.17</td>
      <td>11.63</td>
      <td>5.02</td>
      <td>42.0</td>
      <td>17.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>57.0</td>
      <td>26.0</td>
      <td>12.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>CHC</th>
      <th>NL Central</th>
      <td>54.58</td>
      <td>27.82</td>
      <td>10.47</td>
      <td>4.86</td>
      <td>42.0</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>68.0</td>
      <td>32.0</td>
      <td>11.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>NYY</th>
      <th>AL East</th>
      <td>46.93</td>
      <td>19.44</td>
      <td>9.19</td>
      <td>3.82</td>
      <td>75.0</td>
      <td>33.0</td>
      <td>33.0</td>
      <td>0.0</td>
      <td>54.0</td>
      <td>30.0</td>
      <td>17.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>TOR</th>
      <th>AL East</th>
      <td>36.70</td>
      <td>17.04</td>
      <td>7.78</td>
      <td>3.17</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>31.0</td>
      <td>11.0</td>
      <td>4.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>CIN</th>
      <th>NL Central</th>
      <td>43.22</td>
      <td>20.36</td>
      <td>6.82</td>
      <td>2.56</td>
      <td>83.0</td>
      <td>83.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>42.0</td>
      <td>23.0</td>
      <td>7.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>MIA</th>
      <th>NL East</th>
      <td>45.42</td>
      <td>20.18</td>
      <td>6.53</td>
      <td>2.47</td>
      <td>50.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>32.0</td>
      <td>10.0</td>
      <td>2.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>STL</th>
      <th>NL Central</th>
      <td>38.11</td>
      <td>12.33</td>
      <td>5.84</td>
      <td>2.34</td>
      <td>17.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>41.0</td>
      <td>12.0</td>
      <td>6.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>HOU</th>
      <th>AL West</th>
      <td>36.17</td>
      <td>13.53</td>
      <td>4.48</td>
      <td>1.41</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>45.0</td>
      <td>25.0</td>
      <td>12.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>MIL</th>
      <th>NL Central</th>
      <td>26.14</td>
      <td>9.79</td>
      <td>3.84</td>
      <td>1.17</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>23.0</td>
      <td>11.0</td>
      <td>5.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>SEA</th>
      <th>AL West</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>SF</th>
      <th>NL West</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>PIT</th>
      <th>NL Central</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TEX</th>
      <th>AL West</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>ARI</th>
      <th>NL West</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>PHI</th>
      <th>NL East</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>NYM</th>
      <th>NL East</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>LAA</th>
      <th>AL West</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>KC</th>
      <th>AL Central</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>DET</th>
      <th>AL Central</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>COL</th>
      <th>NL West</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>BOS</th>
      <th>AL East</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>BAL</th>
      <th>AL East</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>WSH</th>
      <th>NL East</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



## The model

The model is relatively straightforward. For each team, we learn a latent factor, $a_{t}$, representing their ability to win based on the outcomes of every series in the 2020 regular season. Technically this is a probabilistic model, so the latent factors are really distributions. The distributions are fit such that


$$
\begin{equation}
\frac{\text{exp}(a_{t1})}{\text{exp}(a_{t_{1}}) + \text{exp}(a_{t_{2}})}
\end{equation}
$$

represents the probability that team $t_{1}$ will beat $t_{2}$ in a given game. The code looks like

```python
with pm.Model() as model:
    σ_a = pm.Exponential('σ_a', np.log(5))

    a_t = pm.Normal('a_t', mu=0, sigma=σ_a, shape=n_teams)
    a_1, a_2 = a_t[home_team_id], a_t[away_team_id]
    a = T.stack([a_1, a_2]).T

    p = pm.Deterministic('p', softmax(a))
    wins = pm.Binomial('wins', n=n_matchups, p=p, shape=(n_games, 2), observed=observed_wins)

    trace = pm.sample(5_000, tune=4_000)
```

Once the  model is fit we can sample from each team's distributions (more precisely,  we sample from  the samples approximating the posterior) and  simulate a postseason outcome. After doing this a bunch  we end up with probabilities  for each team's postseason success.

For full details you can find the notebook used  to generate the  projections  on my GitHub below or read up  on the earlier  blog posts I linked above.

One notable difference between this model and the 2018 model  is that home team advantage  is  not accounted  for. While playing  games on the  road versus at  home certainly had  an impact this season, all postseason games [are being played on neutral  sites](https://www.mlb.com/news/mlb-2020-postseason-schedule-announced) so  we left this feature out. Another change is that the [softmax function](https://en.wikipedia.org/wiki/Softmax_function) is used to calculate  the probability  of winning  as  opposed to  the [Dirichlet distribution](https://en.wikipedia.org/wiki/Dirichlet_distribution). The methods are  the  same, except  the softmax uses an exponential transform which allows the distributions to  include negative values  and results in more consistent distributions, regardless of team ability (compare the team  quality estimates below to my earlier post).

As before the teams can be ranked according to the learned distributions.



![png]({{ "/assets/mlb-2020-postseason-projections/output_5_0.png" | asbolute_url }})