
<center><img src="https://storage.googleapis.com/www-theplayerstribune-com/uploads/GettyImages-492684424-1.jpg" width="836" height="400"></center>
<h1><center>Predicting MLB Teams' Regular Season Win Totals</center></h1>
<h4><center>CMSC320 Final Tutorial by Jason Schneider</center></h4>
<hr>
<h2><center>An Introduction</center></h2>
<p>
    Predicting the outcomes of sporting events has been a hobby of sports fan and analysts for nearly as long
    as sport itself.  Predicting sports outcomes is a crucial element of popular sports television like ESPN,
    sports gambling, fantasy sports, and coaching purposes.  Out of all sports today, baseball, namely MLB
    (Major League Baseball), is currently the statistically-driven sport in America.  This is thanks to the
    statistical revolution started by Bill James and was popularized by the Oakland Athletics' General Manager: Billy
    Beane.  We explored Billy Beane and his application of data science and sabermetrical analysis in Project 2
    earlier this semester, but the applications of sabermetrical analysis of baseball go beyond confirming a bias
    between poor and rich teams.  We can use it to create a predictive model for various questions as well.  The
    question we want to answer here is: <strong>How can we accurately predict the number of regular season wins every
    MLB team will have next season?
    </strong>
    <br><br>
    Luckily, with the advancements in computational technologies, this and many more questions can be answered to
    various degrees of accuracy.  While there are many, many methods and models out there that all argue they are
    the "most accurate," claim to "involve the latest trends in statistics," and "the most up-to-date techniques,"
    there is no mathematical model that can 100% accurately predict every result.  There's an old saying that
    <i>anything can happen in baseball</i>.  What we hope to accomplish here is find a way to get as close to 100%
    accuracy as possible.  In this tutorial, we will divise our own methods, statistics, and trends to base our model
    off of, different from those that already exist so we can use our own research to try to create the most accurate
    model possible, but also more simple so that it is easier for readers of this tutorial to follow along.
</p>
<hr>
<h2><center>Table of Contents</center></h2>
<ul>
    <li><strong>1 -> Getting Started</strong>
        <ul>
            <li>1.1 -> <i>Required Libraries</i></li>
            <li>1.2 -> <i>Data Sources Used & Data Required</i></li>
            <li>1.3 -> <i>Scraping, Loading, and Formatting Data</i></li>
            <li>1.4 -> <i>End Goal Product</i></li>
        </ul>
    </li>
    <li><strong>2 -> What Statistics Contribute to More Wins?</strong>
        <ul>
            <li>2.1 -> <i>Choosing a Starting Point</i></li>
            <li>2.2 -> <i>Example of Relationship Analyzation: Batting Average!</i></li>
            <li>2.3 -> <i>Relationship: Runs Scored and Wins</i></li>
            <li>2.4 -> <i>Relationship: WHIP and Wins</i></li>
            <li>2.5 -> <i>Relationship: WAR and Wins</i></li>
        </ul>
    </li>
    <li><strong>3 -> Putting It All Together</strong>
</ul>
<hr>

<h2><center>1 Getting Started</center></h2>

<h3>1.1 Required Libraries</h3>
<p>
    The required import libraries and matplotlib settings for this project are shown below with a brief description
    of their purpose (text in parenthesis represents alias used for import):
</p>
<ul>
    <li><b>Math <small>(math)</small>: </b>General math-based functionality not included in vanilla Python 3</li>
    <li><b>Regular Expressions <small>(re)</small>: </b>Regular Expression and replacement operations</li>
    <li><b>Requests <small>(rq)</small>: </b>GET Requests to data source sites</li>
    <li><b>Pandas <small>(pd)</small>: </b>Scraping, Manipulating, displaying, and formatting data</li>
    <li><b>Numpy <small>(np)</small>: </b>Sabermetrical analysis and aiding in matrix use</li>
    <li><b>Matplotlib's Pyplot <small>(pypt)</small>: </b>Basic graphing, plotting, and data analysis</li>
    <li><b>Seaborn <small>(sea)</small>: </b>More robust graphing, plotting, and modelling tools</li>
    <li><b>scikit-learn's linear_model <small>(lmod)</small>: </b>Aids in creating linear models involved in
    tutorial</li>
    <li><b>BeautifulSoup4 <small>(bsoup)</small>: </b>Scraping and importing data for use with Pandas</li>
    <li><b>Scipy's Pearson Correlation <small>(pr)</small>: </b>Validating relationships between certain
    stats/attributes and wins</li>
    <li><b>Statsmodels's api <small>(sm)</small>: </b>Performing Multiple Linear Regression in our "research"</li>
    <li><i>%matplotlib inline: </i>Ensure that any Matplotlib plots are printed inline for notebook</li>
</ul>


```python
# Imports for all required libraries:

import math
import re
import requests as rq
import pandas as pd
import numpy as np
import matplotlib.pyplot as pypt
import seaborn as sea
from sklearn import linear_model as lmod
from bs4 import BeautifulSoup as bsoup
from scipy.stats import pearsonr as pr
from statsmodels import api as sm
%matplotlib inline
```

<h3>1.2 Data Source Used & Data Required</h3>
<h4><center>Data Source Used</center></h4>
<p>
    The data source that we will be using is: <b><i><a href="https://www.baseball-reference.com/">Baseball
    Reference</a></i></b>.
    <br><br>
    <b><i>Baseball Reference</i></b> is a fantastic website and robust, expansive statistics data source for a 
    wide range of baseball fans: from those who just picked their favorite team to those who want the edge in all 10 
    of their fantasy baseball leagues.  In addition, it is very friendly to data scraping and has a clear, usable 
    HTML DOM structure.  In a sense, it can be considered the <i>Baseball Wikipedia</i>.
    <br><br>
    Baseball Reference was launched in 2000 by Sean Forman, a Mathematics Professor at St. Joseph's University and
    baseball researcher.
</p>
<br>
<h4><center>Data Required</center></h4>
<p>
    The data involved in predicting season win totals varies between every implementation of this idea.  Why?
    <strong>Briefly, every implementation places different emphasis of contrubtion for different 
    correlations.</strong>  Our basic model will look much different than by companies who spend millions of dollars
    into researching but that doesn't mean we can't make an effective predictive model without doing so.
    <br><br>
    While the actual specific data within these categories will be explored and explained later, the types of data we 
    will be interested in retrieving are:
    <ul>
        <li><b>Constant Data:</b>
            <ul>
                <li>Team Names</li>
                <li>Divisions</li>
                <li>Leagues</li>
            </ul>
        </li>
        <li><b><i>n</i> Previous Seasons:</b>
            <ul>
                <li>Win-Loss Record and Ratio</li>
                <li>Regular Season Game Logs, containing things like (but not limited too)
                    <ul>
                        <li>Box Score</li>
                        <li>Game Time and Length</li>
                        <li>Weather</li>
                        <li>Player Performances</li>
                        <li><i>etc...</i></li>
                    </ul>
                </li>
                <li>Team Performance Statistics</li>
                <li>Player Performance Statistics</li>
            </ul>
        </li>
    </ul>
    <br>
</p>
<h4><center>Things to Keep In Mind</center></h4>
<p>
    We might not end up using all of this information if we find in our investigations that they are not needed, but
    this is a good general starting list for really any sport: Baseball, Football, Basketball, Hockey, Soccer, you
    name it.
    <br><br>
    <ul>
        <i>
        <li>For this tutorial, we will be using data from the last <b>n = 15</b> seasons by default, though more 
            or less data may make the resulting model more or less accurate.</li>
        <li>For the purposes of this tutorial, the divisions and leagues will be based offof the team names,
            divisions, and leagues as of the end of the 2019 MLB season.  That means that the change of Tampa Bay's
            name to the Rays and the Astros' move to the American League in 2013 are treated as if they always
            existed that way.</li>
        <li>Predictions and Past Data assumes that <b>162 Games</b> will be or were played, though this can change in
            rare cases like players union holdouts and tie-breaker games, which we will ignore as these are rather up
            to chance and are not up to our model to predict.</li>
        </i>
    </ul>
</p>

<h3>1.3 Scraping, Loading, and Formatting the Data</h3>
<p>
    Now it's time to get all of the juicy, raw data we want from Baseball Reference into usable, cleanly formatted
    pandas dataframes.  In order to do this, we need to make use of <code>requests</code>,
    <code>bs4</code> (BeautifulSoup4), and <code>pandas</code>.  We need to:
    <ol>
        <li>Use <b>requests</b> to: Send <b>GET</b> requests to Baseball Reference for multiple Baseball Reference
        pages <a href="https://www.baseball-reference.com/leagues/MLB/">MLB Team Win Totals</a> and MLB Season by
        Season Data for 2005 to 2019 seasons <a href="https://www.baseball-reference.com/leagues/MLB/2019.shtml">
        (2019 version as example)</a></li>
        <li>Use <b>bs4</b> to: Find and store table elements</li>
        <li>Use <b>pandas</b> to: Read html table elements into pandas dataframes, then to reformat and display data
    </ol>
    <br>
    We'll run through these steps with the constant data first as this is more straightforward and the we'll repeat
    these steps for past seasons data as it involves more steps.
</p>


```python
# Dealing with the "Constant Data" from the MLB Team Win Totals page #

# GET Request for MLB Team Win Totals Page
req_teamwintotals = rq.get("https://www.baseball-reference.com/leagues/MLB/")

# Parses request as HTML and finds and stores the desired table element
root_teamwintotals = bsoup(req_teamwintotals.content, "html")
table_teamwintotals = root_teamwintotals.find("table")

# Read the html element then create dataframe.  We will use all pre-existing column names (as they are in a great
# format already) unless otherwise shown in code.
pdtable_twt = pd.read_html(str(table_teamwintotals))
df_twt = pdtable_twt[0]

# Our new dataframe looks like:
df_twt
```




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
      <th>Year</th>
      <th>G</th>
      <th>ARI</th>
      <th>ATL</th>
      <th>BLA</th>
      <th>BAL</th>
      <th>BOS</th>
      <th>CHC</th>
      <th>CHW</th>
      <th>CIN</th>
      <th>...</th>
      <th>PHI</th>
      <th>PIT</th>
      <th>SDP</th>
      <th>SFG</th>
      <th>SEA</th>
      <th>STL</th>
      <th>TBR</th>
      <th>TEX</th>
      <th>TOR</th>
      <th>WSN</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2019</td>
      <td>162</td>
      <td>85</td>
      <td>97</td>
      <td>NaN</td>
      <td>54</td>
      <td>84</td>
      <td>84</td>
      <td>72</td>
      <td>75</td>
      <td>...</td>
      <td>81</td>
      <td>69</td>
      <td>70</td>
      <td>77</td>
      <td>68</td>
      <td>91</td>
      <td>96</td>
      <td>78</td>
      <td>67</td>
      <td>93</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2018</td>
      <td>163</td>
      <td>82</td>
      <td>90</td>
      <td>NaN</td>
      <td>47</td>
      <td>108</td>
      <td>95</td>
      <td>62</td>
      <td>67</td>
      <td>...</td>
      <td>80</td>
      <td>82</td>
      <td>66</td>
      <td>73</td>
      <td>89</td>
      <td>88</td>
      <td>90</td>
      <td>67</td>
      <td>73</td>
      <td>82</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2017</td>
      <td>162</td>
      <td>93</td>
      <td>72</td>
      <td>NaN</td>
      <td>75</td>
      <td>93</td>
      <td>92</td>
      <td>67</td>
      <td>68</td>
      <td>...</td>
      <td>66</td>
      <td>75</td>
      <td>71</td>
      <td>64</td>
      <td>78</td>
      <td>83</td>
      <td>80</td>
      <td>78</td>
      <td>76</td>
      <td>97</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2016</td>
      <td>162</td>
      <td>69</td>
      <td>68</td>
      <td>NaN</td>
      <td>89</td>
      <td>93</td>
      <td>103</td>
      <td>78</td>
      <td>68</td>
      <td>...</td>
      <td>71</td>
      <td>78</td>
      <td>68</td>
      <td>87</td>
      <td>86</td>
      <td>86</td>
      <td>68</td>
      <td>95</td>
      <td>89</td>
      <td>95</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2015</td>
      <td>162</td>
      <td>79</td>
      <td>67</td>
      <td>NaN</td>
      <td>81</td>
      <td>78</td>
      <td>97</td>
      <td>76</td>
      <td>64</td>
      <td>...</td>
      <td>63</td>
      <td>98</td>
      <td>74</td>
      <td>84</td>
      <td>76</td>
      <td>100</td>
      <td>80</td>
      <td>88</td>
      <td>93</td>
      <td>83</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2014</td>
      <td>162</td>
      <td>64</td>
      <td>79</td>
      <td>NaN</td>
      <td>96</td>
      <td>71</td>
      <td>73</td>
      <td>73</td>
      <td>76</td>
      <td>...</td>
      <td>73</td>
      <td>88</td>
      <td>77</td>
      <td>88</td>
      <td>87</td>
      <td>90</td>
      <td>77</td>
      <td>67</td>
      <td>83</td>
      <td>96</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2013</td>
      <td>163</td>
      <td>81</td>
      <td>96</td>
      <td>NaN</td>
      <td>85</td>
      <td>97</td>
      <td>66</td>
      <td>63</td>
      <td>90</td>
      <td>...</td>
      <td>73</td>
      <td>94</td>
      <td>76</td>
      <td>76</td>
      <td>71</td>
      <td>97</td>
      <td>92</td>
      <td>91</td>
      <td>74</td>
      <td>86</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2012</td>
      <td>162</td>
      <td>81</td>
      <td>94</td>
      <td>NaN</td>
      <td>93</td>
      <td>69</td>
      <td>61</td>
      <td>85</td>
      <td>97</td>
      <td>...</td>
      <td>81</td>
      <td>79</td>
      <td>76</td>
      <td>94</td>
      <td>75</td>
      <td>88</td>
      <td>90</td>
      <td>93</td>
      <td>73</td>
      <td>98</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2011</td>
      <td>162</td>
      <td>94</td>
      <td>89</td>
      <td>NaN</td>
      <td>69</td>
      <td>90</td>
      <td>71</td>
      <td>79</td>
      <td>79</td>
      <td>...</td>
      <td>102</td>
      <td>72</td>
      <td>71</td>
      <td>86</td>
      <td>67</td>
      <td>90</td>
      <td>91</td>
      <td>96</td>
      <td>81</td>
      <td>80</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2010</td>
      <td>162</td>
      <td>65</td>
      <td>91</td>
      <td>NaN</td>
      <td>66</td>
      <td>89</td>
      <td>75</td>
      <td>88</td>
      <td>91</td>
      <td>...</td>
      <td>97</td>
      <td>57</td>
      <td>90</td>
      <td>92</td>
      <td>61</td>
      <td>86</td>
      <td>96</td>
      <td>90</td>
      <td>85</td>
      <td>69</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2009</td>
      <td>163</td>
      <td>70</td>
      <td>86</td>
      <td>NaN</td>
      <td>64</td>
      <td>95</td>
      <td>83</td>
      <td>79</td>
      <td>78</td>
      <td>...</td>
      <td>93</td>
      <td>62</td>
      <td>75</td>
      <td>88</td>
      <td>85</td>
      <td>91</td>
      <td>84</td>
      <td>87</td>
      <td>75</td>
      <td>59</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2008</td>
      <td>163</td>
      <td>82</td>
      <td>72</td>
      <td>NaN</td>
      <td>68</td>
      <td>95</td>
      <td>97</td>
      <td>89</td>
      <td>74</td>
      <td>...</td>
      <td>92</td>
      <td>67</td>
      <td>63</td>
      <td>72</td>
      <td>61</td>
      <td>86</td>
      <td>97</td>
      <td>79</td>
      <td>86</td>
      <td>59</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2007</td>
      <td>163</td>
      <td>90</td>
      <td>84</td>
      <td>NaN</td>
      <td>69</td>
      <td>96</td>
      <td>85</td>
      <td>72</td>
      <td>72</td>
      <td>...</td>
      <td>89</td>
      <td>68</td>
      <td>89</td>
      <td>71</td>
      <td>88</td>
      <td>78</td>
      <td>66</td>
      <td>75</td>
      <td>83</td>
      <td>73</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2006</td>
      <td>162</td>
      <td>76</td>
      <td>79</td>
      <td>NaN</td>
      <td>70</td>
      <td>86</td>
      <td>66</td>
      <td>90</td>
      <td>80</td>
      <td>...</td>
      <td>85</td>
      <td>67</td>
      <td>88</td>
      <td>76</td>
      <td>78</td>
      <td>83</td>
      <td>61</td>
      <td>80</td>
      <td>87</td>
      <td>71</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2005</td>
      <td>162</td>
      <td>77</td>
      <td>90</td>
      <td>NaN</td>
      <td>74</td>
      <td>95</td>
      <td>79</td>
      <td>99</td>
      <td>73</td>
      <td>...</td>
      <td>88</td>
      <td>67</td>
      <td>82</td>
      <td>75</td>
      <td>69</td>
      <td>100</td>
      <td>67</td>
      <td>79</td>
      <td>80</td>
      <td>81</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2004</td>
      <td>162</td>
      <td>51</td>
      <td>96</td>
      <td>NaN</td>
      <td>78</td>
      <td>98</td>
      <td>89</td>
      <td>83</td>
      <td>76</td>
      <td>...</td>
      <td>86</td>
      <td>72</td>
      <td>87</td>
      <td>91</td>
      <td>63</td>
      <td>105</td>
      <td>70</td>
      <td>89</td>
      <td>67</td>
      <td>67</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2003</td>
      <td>162</td>
      <td>84</td>
      <td>101</td>
      <td>NaN</td>
      <td>71</td>
      <td>95</td>
      <td>88</td>
      <td>86</td>
      <td>69</td>
      <td>...</td>
      <td>86</td>
      <td>75</td>
      <td>64</td>
      <td>100</td>
      <td>93</td>
      <td>85</td>
      <td>63</td>
      <td>71</td>
      <td>86</td>
      <td>83</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2002</td>
      <td>162</td>
      <td>98</td>
      <td>101</td>
      <td>NaN</td>
      <td>67</td>
      <td>93</td>
      <td>67</td>
      <td>81</td>
      <td>78</td>
      <td>...</td>
      <td>80</td>
      <td>72</td>
      <td>66</td>
      <td>95</td>
      <td>93</td>
      <td>97</td>
      <td>55</td>
      <td>72</td>
      <td>78</td>
      <td>83</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2001</td>
      <td>162</td>
      <td>92</td>
      <td>88</td>
      <td>NaN</td>
      <td>63</td>
      <td>82</td>
      <td>88</td>
      <td>83</td>
      <td>66</td>
      <td>...</td>
      <td>86</td>
      <td>62</td>
      <td>79</td>
      <td>90</td>
      <td>116</td>
      <td>93</td>
      <td>62</td>
      <td>73</td>
      <td>80</td>
      <td>68</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2000</td>
      <td>162</td>
      <td>85</td>
      <td>95</td>
      <td>NaN</td>
      <td>74</td>
      <td>85</td>
      <td>65</td>
      <td>95</td>
      <td>85</td>
      <td>...</td>
      <td>65</td>
      <td>69</td>
      <td>76</td>
      <td>97</td>
      <td>91</td>
      <td>95</td>
      <td>69</td>
      <td>71</td>
      <td>83</td>
      <td>67</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1999</td>
      <td>163</td>
      <td>100</td>
      <td>103</td>
      <td>NaN</td>
      <td>78</td>
      <td>94</td>
      <td>67</td>
      <td>75</td>
      <td>96</td>
      <td>...</td>
      <td>77</td>
      <td>78</td>
      <td>74</td>
      <td>86</td>
      <td>79</td>
      <td>75</td>
      <td>69</td>
      <td>95</td>
      <td>84</td>
      <td>68</td>
    </tr>
    <tr>
      <th>21</th>
      <td>1998</td>
      <td>163</td>
      <td>65</td>
      <td>106</td>
      <td>NaN</td>
      <td>79</td>
      <td>92</td>
      <td>90</td>
      <td>80</td>
      <td>77</td>
      <td>...</td>
      <td>75</td>
      <td>69</td>
      <td>98</td>
      <td>89</td>
      <td>76</td>
      <td>83</td>
      <td>63</td>
      <td>88</td>
      <td>88</td>
      <td>65</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1997</td>
      <td>162</td>
      <td>NaN</td>
      <td>101</td>
      <td>NaN</td>
      <td>98</td>
      <td>78</td>
      <td>68</td>
      <td>80</td>
      <td>76</td>
      <td>...</td>
      <td>68</td>
      <td>79</td>
      <td>76</td>
      <td>90</td>
      <td>90</td>
      <td>73</td>
      <td>NaN</td>
      <td>77</td>
      <td>76</td>
      <td>78</td>
    </tr>
    <tr>
      <th>23</th>
      <td>1996</td>
      <td>162</td>
      <td>NaN</td>
      <td>96</td>
      <td>NaN</td>
      <td>88</td>
      <td>85</td>
      <td>76</td>
      <td>85</td>
      <td>81</td>
      <td>...</td>
      <td>67</td>
      <td>73</td>
      <td>91</td>
      <td>68</td>
      <td>85</td>
      <td>88</td>
      <td>NaN</td>
      <td>90</td>
      <td>74</td>
      <td>88</td>
    </tr>
    <tr>
      <th>24</th>
      <td>1995</td>
      <td>145</td>
      <td>NaN</td>
      <td>90</td>
      <td>NaN</td>
      <td>71</td>
      <td>86</td>
      <td>73</td>
      <td>68</td>
      <td>85</td>
      <td>...</td>
      <td>69</td>
      <td>58</td>
      <td>70</td>
      <td>67</td>
      <td>79</td>
      <td>62</td>
      <td>NaN</td>
      <td>74</td>
      <td>56</td>
      <td>66</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Year</td>
      <td>G</td>
      <td>ARI</td>
      <td>ATL</td>
      <td>BLA</td>
      <td>BAL</td>
      <td>BOS</td>
      <td>CHC</td>
      <td>CHW</td>
      <td>CIN</td>
      <td>...</td>
      <td>PHI</td>
      <td>PIT</td>
      <td>SDP</td>
      <td>SFG</td>
      <td>SEA</td>
      <td>STL</td>
      <td>TBR</td>
      <td>TEX</td>
      <td>TOR</td>
      <td>WSN</td>
    </tr>
    <tr>
      <th>26</th>
      <td>1994</td>
      <td>117</td>
      <td>NaN</td>
      <td>68</td>
      <td>NaN</td>
      <td>63</td>
      <td>54</td>
      <td>49</td>
      <td>67</td>
      <td>66</td>
      <td>...</td>
      <td>54</td>
      <td>53</td>
      <td>47</td>
      <td>55</td>
      <td>49</td>
      <td>53</td>
      <td>NaN</td>
      <td>52</td>
      <td>55</td>
      <td>74</td>
    </tr>
    <tr>
      <th>27</th>
      <td>1993</td>
      <td>162</td>
      <td>NaN</td>
      <td>104</td>
      <td>NaN</td>
      <td>85</td>
      <td>80</td>
      <td>84</td>
      <td>94</td>
      <td>73</td>
      <td>...</td>
      <td>97</td>
      <td>75</td>
      <td>61</td>
      <td>103</td>
      <td>82</td>
      <td>87</td>
      <td>NaN</td>
      <td>86</td>
      <td>95</td>
      <td>94</td>
    </tr>
    <tr>
      <th>28</th>
      <td>1992</td>
      <td>162</td>
      <td>NaN</td>
      <td>98</td>
      <td>NaN</td>
      <td>89</td>
      <td>73</td>
      <td>78</td>
      <td>86</td>
      <td>90</td>
      <td>...</td>
      <td>70</td>
      <td>96</td>
      <td>82</td>
      <td>72</td>
      <td>64</td>
      <td>83</td>
      <td>NaN</td>
      <td>77</td>
      <td>96</td>
      <td>87</td>
    </tr>
    <tr>
      <th>29</th>
      <td>1991</td>
      <td>162</td>
      <td>NaN</td>
      <td>94</td>
      <td>NaN</td>
      <td>67</td>
      <td>84</td>
      <td>77</td>
      <td>87</td>
      <td>74</td>
      <td>...</td>
      <td>78</td>
      <td>98</td>
      <td>84</td>
      <td>75</td>
      <td>83</td>
      <td>84</td>
      <td>NaN</td>
      <td>85</td>
      <td>91</td>
      <td>71</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>93</th>
      <td>1929</td>
      <td>154</td>
      <td>NaN</td>
      <td>56</td>
      <td>NaN</td>
      <td>79</td>
      <td>58</td>
      <td>98</td>
      <td>59</td>
      <td>66</td>
      <td>...</td>
      <td>71</td>
      <td>88</td>
      <td>NaN</td>
      <td>84</td>
      <td>NaN</td>
      <td>78</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>94</th>
      <td>1928</td>
      <td>154</td>
      <td>NaN</td>
      <td>50</td>
      <td>NaN</td>
      <td>82</td>
      <td>57</td>
      <td>91</td>
      <td>72</td>
      <td>78</td>
      <td>...</td>
      <td>43</td>
      <td>85</td>
      <td>NaN</td>
      <td>93</td>
      <td>NaN</td>
      <td>95</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>95</th>
      <td>1927</td>
      <td>154</td>
      <td>NaN</td>
      <td>60</td>
      <td>NaN</td>
      <td>59</td>
      <td>51</td>
      <td>85</td>
      <td>70</td>
      <td>75</td>
      <td>...</td>
      <td>51</td>
      <td>94</td>
      <td>NaN</td>
      <td>92</td>
      <td>NaN</td>
      <td>92</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>96</th>
      <td>1926</td>
      <td>154</td>
      <td>NaN</td>
      <td>66</td>
      <td>NaN</td>
      <td>62</td>
      <td>46</td>
      <td>82</td>
      <td>81</td>
      <td>87</td>
      <td>...</td>
      <td>58</td>
      <td>84</td>
      <td>NaN</td>
      <td>74</td>
      <td>NaN</td>
      <td>89</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>97</th>
      <td>1925</td>
      <td>154</td>
      <td>NaN</td>
      <td>70</td>
      <td>NaN</td>
      <td>82</td>
      <td>47</td>
      <td>68</td>
      <td>79</td>
      <td>80</td>
      <td>...</td>
      <td>68</td>
      <td>95</td>
      <td>NaN</td>
      <td>86</td>
      <td>NaN</td>
      <td>77</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>98</th>
      <td>1924</td>
      <td>154</td>
      <td>NaN</td>
      <td>53</td>
      <td>NaN</td>
      <td>74</td>
      <td>67</td>
      <td>81</td>
      <td>66</td>
      <td>83</td>
      <td>...</td>
      <td>55</td>
      <td>90</td>
      <td>NaN</td>
      <td>93</td>
      <td>NaN</td>
      <td>65</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>99</th>
      <td>1923</td>
      <td>154</td>
      <td>NaN</td>
      <td>54</td>
      <td>NaN</td>
      <td>74</td>
      <td>61</td>
      <td>83</td>
      <td>69</td>
      <td>91</td>
      <td>...</td>
      <td>50</td>
      <td>87</td>
      <td>NaN</td>
      <td>95</td>
      <td>NaN</td>
      <td>79</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>100</th>
      <td>1922</td>
      <td>154</td>
      <td>NaN</td>
      <td>53</td>
      <td>NaN</td>
      <td>93</td>
      <td>61</td>
      <td>80</td>
      <td>77</td>
      <td>86</td>
      <td>...</td>
      <td>57</td>
      <td>85</td>
      <td>NaN</td>
      <td>93</td>
      <td>NaN</td>
      <td>85</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>101</th>
      <td>1921</td>
      <td>154</td>
      <td>NaN</td>
      <td>79</td>
      <td>NaN</td>
      <td>81</td>
      <td>75</td>
      <td>64</td>
      <td>62</td>
      <td>70</td>
      <td>...</td>
      <td>51</td>
      <td>90</td>
      <td>NaN</td>
      <td>94</td>
      <td>NaN</td>
      <td>87</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>102</th>
      <td>1920</td>
      <td>154</td>
      <td>NaN</td>
      <td>62</td>
      <td>NaN</td>
      <td>76</td>
      <td>72</td>
      <td>75</td>
      <td>96</td>
      <td>82</td>
      <td>...</td>
      <td>62</td>
      <td>79</td>
      <td>NaN</td>
      <td>86</td>
      <td>NaN</td>
      <td>75</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>103</th>
      <td>Year</td>
      <td>G</td>
      <td>ARI</td>
      <td>ATL</td>
      <td>BLA</td>
      <td>BAL</td>
      <td>BOS</td>
      <td>CHC</td>
      <td>CHW</td>
      <td>CIN</td>
      <td>...</td>
      <td>PHI</td>
      <td>PIT</td>
      <td>SDP</td>
      <td>SFG</td>
      <td>SEA</td>
      <td>STL</td>
      <td>TBR</td>
      <td>TEX</td>
      <td>TOR</td>
      <td>WSN</td>
    </tr>
    <tr>
      <th>104</th>
      <td>1919</td>
      <td>140</td>
      <td>NaN</td>
      <td>57</td>
      <td>NaN</td>
      <td>67</td>
      <td>66</td>
      <td>75</td>
      <td>88</td>
      <td>96</td>
      <td>...</td>
      <td>47</td>
      <td>71</td>
      <td>NaN</td>
      <td>87</td>
      <td>NaN</td>
      <td>54</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>105</th>
      <td>1918</td>
      <td>129</td>
      <td>NaN</td>
      <td>53</td>
      <td>NaN</td>
      <td>58</td>
      <td>75</td>
      <td>84</td>
      <td>57</td>
      <td>68</td>
      <td>...</td>
      <td>55</td>
      <td>65</td>
      <td>NaN</td>
      <td>71</td>
      <td>NaN</td>
      <td>51</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>106</th>
      <td>1917</td>
      <td>154</td>
      <td>NaN</td>
      <td>72</td>
      <td>NaN</td>
      <td>57</td>
      <td>90</td>
      <td>74</td>
      <td>100</td>
      <td>78</td>
      <td>...</td>
      <td>87</td>
      <td>51</td>
      <td>NaN</td>
      <td>98</td>
      <td>NaN</td>
      <td>82</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>107</th>
      <td>1916</td>
      <td>154</td>
      <td>NaN</td>
      <td>89</td>
      <td>NaN</td>
      <td>79</td>
      <td>91</td>
      <td>67</td>
      <td>89</td>
      <td>60</td>
      <td>...</td>
      <td>91</td>
      <td>65</td>
      <td>NaN</td>
      <td>86</td>
      <td>NaN</td>
      <td>60</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>108</th>
      <td>1915</td>
      <td>154</td>
      <td>NaN</td>
      <td>83</td>
      <td>NaN</td>
      <td>63</td>
      <td>101</td>
      <td>73</td>
      <td>93</td>
      <td>71</td>
      <td>...</td>
      <td>90</td>
      <td>73</td>
      <td>NaN</td>
      <td>69</td>
      <td>NaN</td>
      <td>72</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>109</th>
      <td>1914</td>
      <td>154</td>
      <td>NaN</td>
      <td>94</td>
      <td>NaN</td>
      <td>71</td>
      <td>91</td>
      <td>78</td>
      <td>70</td>
      <td>60</td>
      <td>...</td>
      <td>74</td>
      <td>69</td>
      <td>NaN</td>
      <td>84</td>
      <td>NaN</td>
      <td>81</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>110</th>
      <td>1913</td>
      <td>154</td>
      <td>NaN</td>
      <td>69</td>
      <td>NaN</td>
      <td>57</td>
      <td>79</td>
      <td>88</td>
      <td>78</td>
      <td>64</td>
      <td>...</td>
      <td>88</td>
      <td>78</td>
      <td>NaN</td>
      <td>101</td>
      <td>NaN</td>
      <td>51</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>111</th>
      <td>1912</td>
      <td>154</td>
      <td>NaN</td>
      <td>52</td>
      <td>NaN</td>
      <td>53</td>
      <td>105</td>
      <td>91</td>
      <td>78</td>
      <td>75</td>
      <td>...</td>
      <td>73</td>
      <td>93</td>
      <td>NaN</td>
      <td>103</td>
      <td>NaN</td>
      <td>63</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>112</th>
      <td>1911</td>
      <td>154</td>
      <td>NaN</td>
      <td>44</td>
      <td>NaN</td>
      <td>45</td>
      <td>78</td>
      <td>92</td>
      <td>77</td>
      <td>70</td>
      <td>...</td>
      <td>79</td>
      <td>85</td>
      <td>NaN</td>
      <td>99</td>
      <td>NaN</td>
      <td>75</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>113</th>
      <td>1910</td>
      <td>154</td>
      <td>NaN</td>
      <td>53</td>
      <td>NaN</td>
      <td>47</td>
      <td>81</td>
      <td>104</td>
      <td>68</td>
      <td>75</td>
      <td>...</td>
      <td>78</td>
      <td>86</td>
      <td>NaN</td>
      <td>91</td>
      <td>NaN</td>
      <td>63</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>114</th>
      <td>1909</td>
      <td>153</td>
      <td>NaN</td>
      <td>45</td>
      <td>NaN</td>
      <td>61</td>
      <td>88</td>
      <td>104</td>
      <td>78</td>
      <td>77</td>
      <td>...</td>
      <td>74</td>
      <td>110</td>
      <td>NaN</td>
      <td>92</td>
      <td>NaN</td>
      <td>54</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>115</th>
      <td>1908</td>
      <td>154</td>
      <td>NaN</td>
      <td>63</td>
      <td>NaN</td>
      <td>83</td>
      <td>75</td>
      <td>99</td>
      <td>88</td>
      <td>73</td>
      <td>...</td>
      <td>83</td>
      <td>98</td>
      <td>NaN</td>
      <td>98</td>
      <td>NaN</td>
      <td>49</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>116</th>
      <td>1907</td>
      <td>154</td>
      <td>NaN</td>
      <td>58</td>
      <td>NaN</td>
      <td>69</td>
      <td>59</td>
      <td>107</td>
      <td>87</td>
      <td>66</td>
      <td>...</td>
      <td>83</td>
      <td>91</td>
      <td>NaN</td>
      <td>82</td>
      <td>NaN</td>
      <td>52</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>117</th>
      <td>1906</td>
      <td>154</td>
      <td>NaN</td>
      <td>49</td>
      <td>NaN</td>
      <td>76</td>
      <td>49</td>
      <td>116</td>
      <td>93</td>
      <td>64</td>
      <td>...</td>
      <td>71</td>
      <td>93</td>
      <td>NaN</td>
      <td>96</td>
      <td>NaN</td>
      <td>52</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>118</th>
      <td>1905</td>
      <td>154</td>
      <td>NaN</td>
      <td>51</td>
      <td>NaN</td>
      <td>54</td>
      <td>78</td>
      <td>92</td>
      <td>92</td>
      <td>79</td>
      <td>...</td>
      <td>83</td>
      <td>96</td>
      <td>NaN</td>
      <td>105</td>
      <td>NaN</td>
      <td>58</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>119</th>
      <td>1904</td>
      <td>154</td>
      <td>NaN</td>
      <td>55</td>
      <td>NaN</td>
      <td>65</td>
      <td>95</td>
      <td>93</td>
      <td>89</td>
      <td>88</td>
      <td>...</td>
      <td>52</td>
      <td>87</td>
      <td>NaN</td>
      <td>106</td>
      <td>NaN</td>
      <td>75</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>120</th>
      <td>1903</td>
      <td>140</td>
      <td>NaN</td>
      <td>58</td>
      <td>NaN</td>
      <td>65</td>
      <td>91</td>
      <td>82</td>
      <td>60</td>
      <td>74</td>
      <td>...</td>
      <td>49</td>
      <td>91</td>
      <td>NaN</td>
      <td>84</td>
      <td>NaN</td>
      <td>43</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>121</th>
      <td>1902</td>
      <td>140</td>
      <td>NaN</td>
      <td>73</td>
      <td>50</td>
      <td>78</td>
      <td>77</td>
      <td>68</td>
      <td>74</td>
      <td>70</td>
      <td>...</td>
      <td>56</td>
      <td>103</td>
      <td>NaN</td>
      <td>48</td>
      <td>NaN</td>
      <td>56</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>122</th>
      <td>1901</td>
      <td>140</td>
      <td>NaN</td>
      <td>69</td>
      <td>68</td>
      <td>48</td>
      <td>79</td>
      <td>53</td>
      <td>83</td>
      <td>52</td>
      <td>...</td>
      <td>83</td>
      <td>90</td>
      <td>NaN</td>
      <td>52</td>
      <td>NaN</td>
      <td>76</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>123 rows × 33 columns</p>
</div>




```python
# Now that we have gotten the data, we need to clean it up so we can use it.  We need to remove the unnecessary BLA
# and G (Games Played) column (BLA is a team that only existed around the 1900s and we said G will "always" be 162), 
# replace missing data with np.nan, and limit this data to just the past 15 seasons (2005 to 2019 as of 12/16/19).
# Before we do any of that, we need to remove the repitition of the headers every 20 years.

df_twt = df_twt[df_twt.Year != "Year"]
df_twt = df_twt.drop(columns=["G", "BLA"])
df_twt = df_twt.replace("NaN", np.nan)
df_twt = df_twt.replace("", np.nan)
df_twt["Year"] = pd.to_numeric(df_twt["Year"])
df_twt = df_twt[df_twt["Year"] >= 2005]
df_twt = df_twt[df_twt["Year"] <= 2019]
df_twt = df_twt.sort_values("Year")
df_twt = df_twt.reset_index(drop = True)

# Here is our formatted, cleaned up dataframe for the MLB Team Team Win Totals data
df_twt
```




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
      <th>Year</th>
      <th>ARI</th>
      <th>ATL</th>
      <th>BAL</th>
      <th>BOS</th>
      <th>CHC</th>
      <th>CHW</th>
      <th>CIN</th>
      <th>CLE</th>
      <th>COL</th>
      <th>...</th>
      <th>PHI</th>
      <th>PIT</th>
      <th>SDP</th>
      <th>SFG</th>
      <th>SEA</th>
      <th>STL</th>
      <th>TBR</th>
      <th>TEX</th>
      <th>TOR</th>
      <th>WSN</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2005</td>
      <td>77</td>
      <td>90</td>
      <td>74</td>
      <td>95</td>
      <td>79</td>
      <td>99</td>
      <td>73</td>
      <td>93</td>
      <td>67</td>
      <td>...</td>
      <td>88</td>
      <td>67</td>
      <td>82</td>
      <td>75</td>
      <td>69</td>
      <td>100</td>
      <td>67</td>
      <td>79</td>
      <td>80</td>
      <td>81</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2006</td>
      <td>76</td>
      <td>79</td>
      <td>70</td>
      <td>86</td>
      <td>66</td>
      <td>90</td>
      <td>80</td>
      <td>78</td>
      <td>76</td>
      <td>...</td>
      <td>85</td>
      <td>67</td>
      <td>88</td>
      <td>76</td>
      <td>78</td>
      <td>83</td>
      <td>61</td>
      <td>80</td>
      <td>87</td>
      <td>71</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2007</td>
      <td>90</td>
      <td>84</td>
      <td>69</td>
      <td>96</td>
      <td>85</td>
      <td>72</td>
      <td>72</td>
      <td>96</td>
      <td>90</td>
      <td>...</td>
      <td>89</td>
      <td>68</td>
      <td>89</td>
      <td>71</td>
      <td>88</td>
      <td>78</td>
      <td>66</td>
      <td>75</td>
      <td>83</td>
      <td>73</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2008</td>
      <td>82</td>
      <td>72</td>
      <td>68</td>
      <td>95</td>
      <td>97</td>
      <td>89</td>
      <td>74</td>
      <td>81</td>
      <td>74</td>
      <td>...</td>
      <td>92</td>
      <td>67</td>
      <td>63</td>
      <td>72</td>
      <td>61</td>
      <td>86</td>
      <td>97</td>
      <td>79</td>
      <td>86</td>
      <td>59</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2009</td>
      <td>70</td>
      <td>86</td>
      <td>64</td>
      <td>95</td>
      <td>83</td>
      <td>79</td>
      <td>78</td>
      <td>65</td>
      <td>92</td>
      <td>...</td>
      <td>93</td>
      <td>62</td>
      <td>75</td>
      <td>88</td>
      <td>85</td>
      <td>91</td>
      <td>84</td>
      <td>87</td>
      <td>75</td>
      <td>59</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2010</td>
      <td>65</td>
      <td>91</td>
      <td>66</td>
      <td>89</td>
      <td>75</td>
      <td>88</td>
      <td>91</td>
      <td>69</td>
      <td>83</td>
      <td>...</td>
      <td>97</td>
      <td>57</td>
      <td>90</td>
      <td>92</td>
      <td>61</td>
      <td>86</td>
      <td>96</td>
      <td>90</td>
      <td>85</td>
      <td>69</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2011</td>
      <td>94</td>
      <td>89</td>
      <td>69</td>
      <td>90</td>
      <td>71</td>
      <td>79</td>
      <td>79</td>
      <td>80</td>
      <td>73</td>
      <td>...</td>
      <td>102</td>
      <td>72</td>
      <td>71</td>
      <td>86</td>
      <td>67</td>
      <td>90</td>
      <td>91</td>
      <td>96</td>
      <td>81</td>
      <td>80</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2012</td>
      <td>81</td>
      <td>94</td>
      <td>93</td>
      <td>69</td>
      <td>61</td>
      <td>85</td>
      <td>97</td>
      <td>68</td>
      <td>64</td>
      <td>...</td>
      <td>81</td>
      <td>79</td>
      <td>76</td>
      <td>94</td>
      <td>75</td>
      <td>88</td>
      <td>90</td>
      <td>93</td>
      <td>73</td>
      <td>98</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2013</td>
      <td>81</td>
      <td>96</td>
      <td>85</td>
      <td>97</td>
      <td>66</td>
      <td>63</td>
      <td>90</td>
      <td>92</td>
      <td>74</td>
      <td>...</td>
      <td>73</td>
      <td>94</td>
      <td>76</td>
      <td>76</td>
      <td>71</td>
      <td>97</td>
      <td>92</td>
      <td>91</td>
      <td>74</td>
      <td>86</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2014</td>
      <td>64</td>
      <td>79</td>
      <td>96</td>
      <td>71</td>
      <td>73</td>
      <td>73</td>
      <td>76</td>
      <td>85</td>
      <td>66</td>
      <td>...</td>
      <td>73</td>
      <td>88</td>
      <td>77</td>
      <td>88</td>
      <td>87</td>
      <td>90</td>
      <td>77</td>
      <td>67</td>
      <td>83</td>
      <td>96</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2015</td>
      <td>79</td>
      <td>67</td>
      <td>81</td>
      <td>78</td>
      <td>97</td>
      <td>76</td>
      <td>64</td>
      <td>81</td>
      <td>68</td>
      <td>...</td>
      <td>63</td>
      <td>98</td>
      <td>74</td>
      <td>84</td>
      <td>76</td>
      <td>100</td>
      <td>80</td>
      <td>88</td>
      <td>93</td>
      <td>83</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2016</td>
      <td>69</td>
      <td>68</td>
      <td>89</td>
      <td>93</td>
      <td>103</td>
      <td>78</td>
      <td>68</td>
      <td>94</td>
      <td>75</td>
      <td>...</td>
      <td>71</td>
      <td>78</td>
      <td>68</td>
      <td>87</td>
      <td>86</td>
      <td>86</td>
      <td>68</td>
      <td>95</td>
      <td>89</td>
      <td>95</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2017</td>
      <td>93</td>
      <td>72</td>
      <td>75</td>
      <td>93</td>
      <td>92</td>
      <td>67</td>
      <td>68</td>
      <td>102</td>
      <td>87</td>
      <td>...</td>
      <td>66</td>
      <td>75</td>
      <td>71</td>
      <td>64</td>
      <td>78</td>
      <td>83</td>
      <td>80</td>
      <td>78</td>
      <td>76</td>
      <td>97</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2018</td>
      <td>82</td>
      <td>90</td>
      <td>47</td>
      <td>108</td>
      <td>95</td>
      <td>62</td>
      <td>67</td>
      <td>91</td>
      <td>91</td>
      <td>...</td>
      <td>80</td>
      <td>82</td>
      <td>66</td>
      <td>73</td>
      <td>89</td>
      <td>88</td>
      <td>90</td>
      <td>67</td>
      <td>73</td>
      <td>82</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2019</td>
      <td>85</td>
      <td>97</td>
      <td>54</td>
      <td>84</td>
      <td>84</td>
      <td>72</td>
      <td>75</td>
      <td>93</td>
      <td>71</td>
      <td>...</td>
      <td>81</td>
      <td>69</td>
      <td>70</td>
      <td>77</td>
      <td>68</td>
      <td>91</td>
      <td>96</td>
      <td>78</td>
      <td>67</td>
      <td>93</td>
    </tr>
  </tbody>
</table>
<p>15 rows × 31 columns</p>
</div>




```python
# Dealing with Previous Seasons' data, on the various MLB Season by Season data pages #

# Since we need to get data from multiple seasons and multiple tables from each season, we'll need to make use of
# loops and perform these steps on each table then stitch together and organize this data into a few more manageable
# "master" tables.

# We want to get and create dataframes for: Team Standard Batting, Team Standard Pitching, and the first two columns
# of MLB Wins Above Avg By Position (also known as WAR).  For purposes of this tutorial, we won't worry about fielding
# as WAR is a complex statistic (explained more later) that includes a player's fielding.  WAR will be explained more
# later, but it is a central statistic to our predictions and is something we will analyze thoroughly.

comments = re.compile("<!--|-->") # Need this to remove comments restricting us from scraping all tables
coldiv = []
df_master = pd.DataFrame()

for i in range(0, 33):
        coldiv.append("***")

for year in range(2005, 2020):
    # Send GET request for given year, then use BeautifulSoup to find root HMTL
    req_yearsbs = rq.get("https://www.baseball-reference.com/leagues/MLB/" + str(year) + ".shtml")
    root_yearsbs = bsoup(comments.sub("", str(req_yearsbs.content)), "html")
    alltables_year = root_yearsbs.findAll("table")
    
    # Create Batting Dataframe
    table_yearbatting = alltables_year[1]
    pdtable_yb = pd.read_html(str(table_yearbatting))
    df_yb = pdtable_yb[0]
    
    # Create Pitching Dataframe
    table_yearpitching = alltables_year[2]
    pdtable_yp = pd.read_html(str(table_yearpitching))
    df_yp = pdtable_yp[0]
    
    # Create WAR Dataframe
    table_yearwar = alltables_year[3]
    pdtable_yw = pd.read_html(str(table_yearwar))
    df_yw = pdtable_yw[0]
    
    # Consolidate all data into "season master table," which we then fill all empty cells and existing NaNs with
    # np.nan, remove all duplicate columns that repeat redundant info (renaming columns if necessary), then add
    # placeholder dividing columns so we can things easier for ourselves later.
    
    df_yearmaster = df_yb.join(df_yp, lsuffix = "_b", rsuffix = "_p").join(df_yw, lsuffix = "", rsuffix = "_w")
    df_yearmaster = df_yearmaster.loc[:,~df_yearmaster.columns.duplicated()]
    df_yearmaster = df_yearmaster.rename(columns = {"Tm_b" : "Tm"})
    df_yearmaster = df_yearmaster.drop(columns = ["Tm_p"])
    df_yearmaster.insert(30, "Col_Div1", coldiv)
    df_yearmaster.insert(67, "Col_Div2", coldiv)
    
    # Lastly we need to include the year next to each of these teams so we know what season this data is for
    include_year = []
    for k in range(0, 33):
        include_year.append(year)
    df_yearmaster.insert(0, "Year", include_year)
    
    # Now, if we've already created the dataframe for the first season, 2005, we need to start consolidating this
    # master SEASON dataframe into a master MASTER (aka dataframe showing all 15 seasons) dataframe by adding each
    # season's data to the bottom of the master df_master dataframe
    if year == 2005:
        df_master = df_yearmaster
    else:
        df_master = df_master.append(df_yearmaster)

# Need to make manual replacements to keep team names consistent, by changing Tampa Bay and Miami's modern team
# abbreviations
df_master = df_master.replace("TBD", "TBR")
df_master = df_master.replace("FLA", "MIA")
df_master = df_master.replace("NaN", np.nan)
df_master = df_master.replace("nan", np.nan)
df_master = df_master.replace("", np.nan)
df_master = df_master.reset_index(drop = True)

# This is our new master table of data for the 2005-2019 seasons:
df_master
```




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
      <th>Year</th>
      <th>Tm</th>
      <th>#Bat</th>
      <th>BatAge</th>
      <th>R/G</th>
      <th>G_b</th>
      <th>PA</th>
      <th>AB</th>
      <th>R_b</th>
      <th>H_b</th>
      <th>...</th>
      <th>1B</th>
      <th>2B_w</th>
      <th>3B_w</th>
      <th>SS</th>
      <th>LF</th>
      <th>CF</th>
      <th>RF</th>
      <th>OF (All)</th>
      <th>DH</th>
      <th>PH</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2005</td>
      <td>ARI</td>
      <td>43</td>
      <td>30.3</td>
      <td>4.30</td>
      <td>162</td>
      <td>6327</td>
      <td>5550</td>
      <td>696</td>
      <td>1419</td>
      <td>...</td>
      <td>STL5.8</td>
      <td>PHI5.6</td>
      <td>NYY6.7</td>
      <td>ATL4.6</td>
      <td>PIT2.6</td>
      <td>ATL4.8</td>
      <td>LAA3.0</td>
      <td>ATL6.6</td>
      <td>CLE2.7</td>
      <td>PHI0.1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2005</td>
      <td>ATL</td>
      <td>45</td>
      <td>28.1</td>
      <td>4.75</td>
      <td>162</td>
      <td>6186</td>
      <td>5486</td>
      <td>769</td>
      <td>1453</td>
      <td>...</td>
      <td>CHC5.5</td>
      <td>BAL4.6</td>
      <td>HOU4.1</td>
      <td>BAL3.3</td>
      <td>NYM2.6</td>
      <td>CLE4.2</td>
      <td>LAD2.3</td>
      <td>PHI6.0</td>
      <td>BOS2.5</td>
      <td>TOR0.1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2005</td>
      <td>BAL</td>
      <td>46</td>
      <td>31.5</td>
      <td>4.50</td>
      <td>162</td>
      <td>6134</td>
      <td>5551</td>
      <td>729</td>
      <td>1492</td>
      <td>...</td>
      <td>TEX4.2</td>
      <td>ARI3.0</td>
      <td>NYM2.8</td>
      <td>CLE3.2</td>
      <td>BOS2.0</td>
      <td>PHI2.6</td>
      <td>SDP2.3</td>
      <td>CLE5.4</td>
      <td>NYY0.1</td>
      <td>BOS0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2005</td>
      <td>BOS</td>
      <td>52</td>
      <td>31.3</td>
      <td>5.62</td>
      <td>162</td>
      <td>6403</td>
      <td>5626</td>
      <td>910</td>
      <td>1579</td>
      <td>...</td>
      <td>COL2.9</td>
      <td>STL2.5</td>
      <td>OAK1.9</td>
      <td>OAK2.7</td>
      <td>FLA1.9</td>
      <td>KCR2.5</td>
      <td>MIL2.2</td>
      <td>BOS5.0</td>
      <td>ARI0.0</td>
      <td>CLE0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2005</td>
      <td>CHC</td>
      <td>46</td>
      <td>29.8</td>
      <td>4.34</td>
      <td>162</td>
      <td>6161</td>
      <td>5584</td>
      <td>703</td>
      <td>1506</td>
      <td>...</td>
      <td>NYY1.5</td>
      <td>LAA2.4</td>
      <td>ATL1.7</td>
      <td>PHI2.6</td>
      <td>TBD1.9</td>
      <td>STL2.5</td>
      <td>SEA1.5</td>
      <td>NYM3.5</td>
      <td>MIL0.0</td>
      <td>LAA-0.2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2005</td>
      <td>CHW</td>
      <td>38</td>
      <td>29.3</td>
      <td>4.57</td>
      <td>162</td>
      <td>6146</td>
      <td>5529</td>
      <td>741</td>
      <td>1450</td>
      <td>...</td>
      <td>ARI1.5</td>
      <td>CLE2.3</td>
      <td>BAL1.7</td>
      <td>PIT2.3</td>
      <td>PHI1.9</td>
      <td>BOS1.8</td>
      <td>WSN1.5</td>
      <td>STL2.6</td>
      <td>PHI0.0</td>
      <td>STL-0.3</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2005</td>
      <td>CIN</td>
      <td>46</td>
      <td>28.6</td>
      <td>5.03</td>
      <td>163</td>
      <td>6321</td>
      <td>5565</td>
      <td>820</td>
      <td>1453</td>
      <td>...</td>
      <td>WSN1.2</td>
      <td>OAK2.0</td>
      <td>CHC1.4</td>
      <td>CIN2.0</td>
      <td>CLE1.7</td>
      <td>CIN1.7</td>
      <td>PHI1.5</td>
      <td>SDP2.4</td>
      <td>PIT0.0</td>
      <td>TEX-0.3</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2005</td>
      <td>CLE</td>
      <td>38</td>
      <td>27.5</td>
      <td>4.88</td>
      <td>162</td>
      <td>6255</td>
      <td>5609</td>
      <td>790</td>
      <td>1522</td>
      <td>...</td>
      <td>SEA1.1</td>
      <td>ATL1.9</td>
      <td>ARI1.1</td>
      <td>TBD1.8</td>
      <td>COL0.9</td>
      <td>MIN1.6</td>
      <td>ATL1.5</td>
      <td>CIN2.3</td>
      <td>SDP0.0</td>
      <td>MIL-0.4</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2005</td>
      <td>COL</td>
      <td>54</td>
      <td>27.2</td>
      <td>4.57</td>
      <td>162</td>
      <td>6238</td>
      <td>5542</td>
      <td>740</td>
      <td>1477</td>
      <td>...</td>
      <td>PHI1.1</td>
      <td>DET1.9</td>
      <td>STL1.0</td>
      <td>NYY1.4</td>
      <td>TOR0.6</td>
      <td>CHW1.3</td>
      <td>BOS1.2</td>
      <td>PIT2.1</td>
      <td>SFG0.0</td>
      <td>SDP-0.4</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2005</td>
      <td>DET</td>
      <td>45</td>
      <td>28.5</td>
      <td>4.46</td>
      <td>162</td>
      <td>6136</td>
      <td>5602</td>
      <td>723</td>
      <td>1521</td>
      <td>...</td>
      <td>MIL0.7</td>
      <td>TOR1.8</td>
      <td>DET0.9</td>
      <td>MIL1.3</td>
      <td>CIN0.6</td>
      <td>DET1.2</td>
      <td>NYY0.6</td>
      <td>LAD1.7</td>
      <td>STL0.0</td>
      <td>DET-0.4</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2005</td>
      <td>MIA</td>
      <td>47</td>
      <td>29.7</td>
      <td>4.43</td>
      <td>162</td>
      <td>6214</td>
      <td>5502</td>
      <td>717</td>
      <td>1499</td>
      <td>...</td>
      <td>CHW0.7</td>
      <td>LAD1.8</td>
      <td>MIL0.8</td>
      <td>HOU0.9</td>
      <td>SFG0.5</td>
      <td>NYM0.9</td>
      <td>STL0.0</td>
      <td>TBD1.7</td>
      <td>HOU0.0</td>
      <td>TBD-0.5</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2005</td>
      <td>HOU</td>
      <td>36</td>
      <td>30.2</td>
      <td>4.25</td>
      <td>163</td>
      <td>6139</td>
      <td>5462</td>
      <td>693</td>
      <td>1400</td>
      <td>...</td>
      <td>FLA0.4</td>
      <td>FLA1.7</td>
      <td>CIN0.6</td>
      <td>TEX0.9</td>
      <td>ATL0.3</td>
      <td>TOR0.9</td>
      <td>CIN0.0</td>
      <td>SEA1.2</td>
      <td>FLA0.0</td>
      <td>BAL-0.5</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2005</td>
      <td>KCR</td>
      <td>46</td>
      <td>28.0</td>
      <td>4.33</td>
      <td>162</td>
      <td>6086</td>
      <td>5503</td>
      <td>701</td>
      <td>1445</td>
      <td>...</td>
      <td>NYM0.3</td>
      <td>CHW0.6</td>
      <td>BOS0.6</td>
      <td>STL0.9</td>
      <td>ARI0.2</td>
      <td>LAD0.9</td>
      <td>NYM0.0</td>
      <td>MIL1.1</td>
      <td>ATL0.0</td>
      <td>FLA-0.6</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2005</td>
      <td>LAA</td>
      <td>40</td>
      <td>29.8</td>
      <td>4.70</td>
      <td>162</td>
      <td>6186</td>
      <td>5624</td>
      <td>761</td>
      <td>1520</td>
      <td>...</td>
      <td>DET0.0</td>
      <td>PIT0.3</td>
      <td>SEA0.4</td>
      <td>MIN0.3</td>
      <td>STL0.1</td>
      <td>SDP0.7</td>
      <td>CHW-0.2</td>
      <td>TOR0.8</td>
      <td>CHC0.0</td>
      <td>ATL-0.6</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2005</td>
      <td>LAD</td>
      <td>44</td>
      <td>28.6</td>
      <td>4.23</td>
      <td>162</td>
      <td>6134</td>
      <td>5433</td>
      <td>685</td>
      <td>1374</td>
      <td>...</td>
      <td>HOU-0.1</td>
      <td>HOU-0.1</td>
      <td>LAA0.1</td>
      <td>CHW0.0</td>
      <td>NYY-0.1</td>
      <td>TBD0.7</td>
      <td>ARI-0.3</td>
      <td>DET0.2</td>
      <td>CIN0.0</td>
      <td>OAK-0.7</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2005</td>
      <td>MIL</td>
      <td>39</td>
      <td>28.4</td>
      <td>4.48</td>
      <td>162</td>
      <td>6156</td>
      <td>5448</td>
      <td>726</td>
      <td>1413</td>
      <td>...</td>
      <td>LAA-0.3</td>
      <td>CHC-0.1</td>
      <td>COL-0.1</td>
      <td>COL0.0</td>
      <td>SEA-0.2</td>
      <td>OAK0.7</td>
      <td>TEX-0.4</td>
      <td>CHW0.0</td>
      <td>COL0.0</td>
      <td>CIN-0.8</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2005</td>
      <td>MIN</td>
      <td>37</td>
      <td>27.6</td>
      <td>4.25</td>
      <td>162</td>
      <td>6192</td>
      <td>5564</td>
      <td>688</td>
      <td>1441</td>
      <td>...</td>
      <td>BOS-0.4</td>
      <td>BOS-0.2</td>
      <td>TOR-0.2</td>
      <td>CHC-0.1</td>
      <td>TEX-0.2</td>
      <td>TEX0.6</td>
      <td>HOU-0.4</td>
      <td>TEX0.0</td>
      <td>LAD-0.1</td>
      <td>CHW-0.9</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2005</td>
      <td>NYM</td>
      <td>42</td>
      <td>28.8</td>
      <td>4.46</td>
      <td>162</td>
      <td>6146</td>
      <td>5505</td>
      <td>722</td>
      <td>1421</td>
      <td>...</td>
      <td>OAK-0.6</td>
      <td>SFG-0.3</td>
      <td>SDP-0.3</td>
      <td>LAA-0.2</td>
      <td>DET-0.5</td>
      <td>PIT0.5</td>
      <td>CLE-0.5</td>
      <td>WSN-0.5</td>
      <td>NYM-0.1</td>
      <td>SEA-1.0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2005</td>
      <td>NYY</td>
      <td>51</td>
      <td>32.2</td>
      <td>5.47</td>
      <td>162</td>
      <td>6406</td>
      <td>5624</td>
      <td>886</td>
      <td>1552</td>
      <td>...</td>
      <td>CIN-0.6</td>
      <td>MIL-0.3</td>
      <td>TBD-0.4</td>
      <td>TOR-0.4</td>
      <td>SDP-0.6</td>
      <td>BAL0.2</td>
      <td>DET-0.5</td>
      <td>OAK-0.6</td>
      <td>TOR-0.2</td>
      <td>ARI-1.0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2005</td>
      <td>OAK</td>
      <td>41</td>
      <td>28.6</td>
      <td>4.77</td>
      <td>162</td>
      <td>6275</td>
      <td>5627</td>
      <td>772</td>
      <td>1476</td>
      <td>...</td>
      <td>TOR-0.7</td>
      <td>TEX-0.6</td>
      <td>PIT-0.4</td>
      <td>NYM-0.6</td>
      <td>OAK-0.6</td>
      <td>MIL0.1</td>
      <td>OAK-0.7</td>
      <td>LAA-0.6</td>
      <td>WSN-0.2</td>
      <td>NYM-1.1</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2005</td>
      <td>PHI</td>
      <td>40</td>
      <td>30.0</td>
      <td>4.98</td>
      <td>162</td>
      <td>6345</td>
      <td>5542</td>
      <td>807</td>
      <td>1494</td>
      <td>...</td>
      <td>TBD-0.9</td>
      <td>WSN-0.7</td>
      <td>CHW-0.6</td>
      <td>SFG-0.7</td>
      <td>HOU-1.0</td>
      <td>HOU-0.1</td>
      <td>TOR-0.7</td>
      <td>FLA-1.0</td>
      <td>DET-0.3</td>
      <td>CHC-1.2</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2005</td>
      <td>PIT</td>
      <td>44</td>
      <td>27.4</td>
      <td>4.20</td>
      <td>162</td>
      <td>6221</td>
      <td>5573</td>
      <td>680</td>
      <td>1445</td>
      <td>...</td>
      <td>CLE-1.0</td>
      <td>CIN-0.9</td>
      <td>CLE-0.7</td>
      <td>DET-0.8</td>
      <td>CHW-1.1</td>
      <td>SEA-0.1</td>
      <td>TBD-0.9</td>
      <td>HOU-1.5</td>
      <td>LAA-0.3</td>
      <td>MIN-1.2</td>
    </tr>
    <tr>
      <th>22</th>
      <td>2005</td>
      <td>SDP</td>
      <td>46</td>
      <td>31.0</td>
      <td>4.22</td>
      <td>162</td>
      <td>6271</td>
      <td>5502</td>
      <td>684</td>
      <td>1416</td>
      <td>...</td>
      <td>SFG-1.0</td>
      <td>TBD-1.1</td>
      <td>WSN-1.0</td>
      <td>SDP-1.1</td>
      <td>MIL-1.2</td>
      <td>WSN-0.4</td>
      <td>PIT-1.0</td>
      <td>MIN-1.7</td>
      <td>SEA-0.3</td>
      <td>HOU-1.3</td>
    </tr>
    <tr>
      <th>23</th>
      <td>2005</td>
      <td>SEA</td>
      <td>48</td>
      <td>28.7</td>
      <td>4.31</td>
      <td>162</td>
      <td>6095</td>
      <td>5507</td>
      <td>699</td>
      <td>1408</td>
      <td>...</td>
      <td>LAD-1.1</td>
      <td>SDP-1.2</td>
      <td>PHI-1.0</td>
      <td>BOS-1.4</td>
      <td>LAD-1.5</td>
      <td>SFG-1.4</td>
      <td>FLA-1.1</td>
      <td>KCR-2.2</td>
      <td>TBD-0.6</td>
      <td>PIT-1.4</td>
    </tr>
    <tr>
      <th>24</th>
      <td>2005</td>
      <td>SFG</td>
      <td>50</td>
      <td>32.2</td>
      <td>4.01</td>
      <td>162</td>
      <td>6077</td>
      <td>5462</td>
      <td>649</td>
      <td>1427</td>
      <td>...</td>
      <td>MIN-1.4</td>
      <td>NYY-2.0</td>
      <td>LAD-1.0</td>
      <td>FLA-1.7</td>
      <td>WSN-1.6</td>
      <td>FLA-1.8</td>
      <td>COL-1.4</td>
      <td>SFG-2.5</td>
      <td>MIN-1.0</td>
      <td>WSN-1.6</td>
    </tr>
    <tr>
      <th>25</th>
      <td>2005</td>
      <td>STL</td>
      <td>40</td>
      <td>30.8</td>
      <td>4.97</td>
      <td>162</td>
      <td>6246</td>
      <td>5538</td>
      <td>805</td>
      <td>1494</td>
      <td>...</td>
      <td>SDP-1.4</td>
      <td>NYM-2.4</td>
      <td>MIN-1.6</td>
      <td>ARI-1.9</td>
      <td>LAA-1.7</td>
      <td>LAA-1.9</td>
      <td>MIN-1.5</td>
      <td>ARI-3.0</td>
      <td>BAL-1.2</td>
      <td>NYY-1.6</td>
    </tr>
    <tr>
      <th>26</th>
      <td>2005</td>
      <td>TBR</td>
      <td>43</td>
      <td>27.5</td>
      <td>4.63</td>
      <td>162</td>
      <td>6120</td>
      <td>5552</td>
      <td>750</td>
      <td>1519</td>
      <td>...</td>
      <td>BAL-2.0</td>
      <td>MIN-2.4</td>
      <td>FLA-1.8</td>
      <td>LAD-2.0</td>
      <td>MIN-1.8</td>
      <td>COL-2.5</td>
      <td>SFG-1.6</td>
      <td>COL-3.0</td>
      <td>TEX-1.3</td>
      <td>LAD-1.7</td>
    </tr>
    <tr>
      <th>27</th>
      <td>2005</td>
      <td>TEX</td>
      <td>50</td>
      <td>27.9</td>
      <td>5.34</td>
      <td>162</td>
      <td>6301</td>
      <td>5716</td>
      <td>865</td>
      <td>1528</td>
      <td>...</td>
      <td>KCR-2.3</td>
      <td>COL-2.5</td>
      <td>TEX-2.2</td>
      <td>SEA-2.1</td>
      <td>BAL-2.1</td>
      <td>CHC-2.7</td>
      <td>CHC-2.0</td>
      <td>NYY-3.3</td>
      <td>CHW-1.7</td>
      <td>COL-1.8</td>
    </tr>
    <tr>
      <th>28</th>
      <td>2005</td>
      <td>TOR</td>
      <td>37</td>
      <td>27.9</td>
      <td>4.78</td>
      <td>162</td>
      <td>6233</td>
      <td>5581</td>
      <td>775</td>
      <td>1480</td>
      <td>...</td>
      <td>ATL-2.6</td>
      <td>SEA-2.8</td>
      <td>SFG-2.5</td>
      <td>WSN-2.9</td>
      <td>CHC-2.5</td>
      <td>ARI-2.9</td>
      <td>KCR-2.0</td>
      <td>BAL-3.9</td>
      <td>KCR-1.7</td>
      <td>KCR-2.0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>2005</td>
      <td>WSN</td>
      <td>55</td>
      <td>29.3</td>
      <td>3.94</td>
      <td>162</td>
      <td>6142</td>
      <td>5426</td>
      <td>639</td>
      <td>1367</td>
      <td>...</td>
      <td>PIT-2.9</td>
      <td>KCR-3.6</td>
      <td>KCR-3.0</td>
      <td>KCR-3.2</td>
      <td>KCR-2.7</td>
      <td>NYY-3.8</td>
      <td>BAL-2.0</td>
      <td>CHC-7.2</td>
      <td>OAK-2.6</td>
      <td>SFG-2.2</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>465</th>
      <td>2019</td>
      <td>BOS</td>
      <td>47</td>
      <td>27.3</td>
      <td>5.56</td>
      <td>162</td>
      <td>6475</td>
      <td>5770</td>
      <td>901</td>
      <td>1554</td>
      <td>...</td>
      <td>ATL2.2</td>
      <td>ARI2.3</td>
      <td>COL3.5</td>
      <td>SDP3.5</td>
      <td>TBR1.7</td>
      <td>ARI2.5</td>
      <td>NYY3.8</td>
      <td>MIL5.6</td>
      <td>LAA0.8</td>
      <td>NYY0.4</td>
    </tr>
    <tr>
      <th>466</th>
      <td>2019</td>
      <td>CHC</td>
      <td>52</td>
      <td>27.7</td>
      <td>5.02</td>
      <td>162</td>
      <td>6195</td>
      <td>5461</td>
      <td>814</td>
      <td>1378</td>
      <td>...</td>
      <td>CHC1.9</td>
      <td>HOU1.9</td>
      <td>ATL3.5</td>
      <td>MIN3.0</td>
      <td>LAD1.1</td>
      <td>HOU2.4</td>
      <td>PHI1.9</td>
      <td>LAA4.6</td>
      <td>KCR0.4</td>
      <td>TBR0.2</td>
    </tr>
    <tr>
      <th>467</th>
      <td>2019</td>
      <td>CHW</td>
      <td>47</td>
      <td>27.6</td>
      <td>4.40</td>
      <td>161</td>
      <td>6042</td>
      <td>5529</td>
      <td>708</td>
      <td>1443</td>
      <td>...</td>
      <td>CLE1.2</td>
      <td>LAD1.6</td>
      <td>BOS2.6</td>
      <td>ARI2.8</td>
      <td>CHC0.5</td>
      <td>NYY2.2</td>
      <td>NYM1.5</td>
      <td>WSN4.0</td>
      <td>BOS0.3</td>
      <td>OAK-0.2</td>
    </tr>
    <tr>
      <th>468</th>
      <td>2019</td>
      <td>CIN</td>
      <td>47</td>
      <td>27.8</td>
      <td>4.33</td>
      <td>162</td>
      <td>6100</td>
      <td>5450</td>
      <td>701</td>
      <td>1328</td>
      <td>...</td>
      <td>NYY0.8</td>
      <td>TBR1.3</td>
      <td>NYY2.4</td>
      <td>BOS2.6</td>
      <td>TEX0.3</td>
      <td>WSN1.6</td>
      <td>HOU1.5</td>
      <td>BOS3.5</td>
      <td>LAD0.1</td>
      <td>CLE-0.3</td>
    </tr>
    <tr>
      <th>469</th>
      <td>2019</td>
      <td>CLE</td>
      <td>54</td>
      <td>27.7</td>
      <td>4.75</td>
      <td>162</td>
      <td>6124</td>
      <td>5425</td>
      <td>769</td>
      <td>1354</td>
      <td>...</td>
      <td>PIT0.7</td>
      <td>BAL1.2</td>
      <td>CHW2.3</td>
      <td>CHC2.4</td>
      <td>NYM0.2</td>
      <td>ATL1.4</td>
      <td>MIN1.0</td>
      <td>MIN3.3</td>
      <td>MIL0.1</td>
      <td>STL-0.3</td>
    </tr>
    <tr>
      <th>470</th>
      <td>2019</td>
      <td>COL</td>
      <td>50</td>
      <td>28.2</td>
      <td>5.15</td>
      <td>162</td>
      <td>6288</td>
      <td>5660</td>
      <td>835</td>
      <td>1502</td>
      <td>...</td>
      <td>STL0.5</td>
      <td>TOR1.0</td>
      <td>LAD2.3</td>
      <td>LAD2.4</td>
      <td>OAK0.1</td>
      <td>LAD1.4</td>
      <td>TBR0.8</td>
      <td>OAK3.3</td>
      <td>TEX0.1</td>
      <td>TEX-0.4</td>
    </tr>
    <tr>
      <th>471</th>
      <td>2019</td>
      <td>DET</td>
      <td>53</td>
      <td>27.6</td>
      <td>3.61</td>
      <td>161</td>
      <td>6039</td>
      <td>5549</td>
      <td>582</td>
      <td>1333</td>
      <td>...</td>
      <td>TBR0.5</td>
      <td>MIL0.9</td>
      <td>CIN2.1</td>
      <td>TBR2.1</td>
      <td>STL0.0</td>
      <td>PIT1.2</td>
      <td>OAK0.2</td>
      <td>TBR3.0</td>
      <td>ATL0.1</td>
      <td>ARI-0.4</td>
    </tr>
    <tr>
      <th>472</th>
      <td>2019</td>
      <td>HOU</td>
      <td>45</td>
      <td>29.0</td>
      <td>5.68</td>
      <td>162</td>
      <td>6394</td>
      <td>5613</td>
      <td>920</td>
      <td>1538</td>
      <td>...</td>
      <td>ARI0.3</td>
      <td>LAA0.5</td>
      <td>CHC1.5</td>
      <td>STL1.8</td>
      <td>MIL0.0</td>
      <td>TEX0.9</td>
      <td>CLE0.2</td>
      <td>PHI1.5</td>
      <td>MIA0.0</td>
      <td>SEA-0.5</td>
    </tr>
    <tr>
      <th>473</th>
      <td>2019</td>
      <td>KCR</td>
      <td>51</td>
      <td>27.6</td>
      <td>4.27</td>
      <td>162</td>
      <td>6080</td>
      <td>5496</td>
      <td>691</td>
      <td>1356</td>
      <td>...</td>
      <td>SEA0.3</td>
      <td>PIT0.5</td>
      <td>MIN1.3</td>
      <td>CHW1.8</td>
      <td>PHI-0.2</td>
      <td>STL0.8</td>
      <td>KCR0.2</td>
      <td>ATL0.9</td>
      <td>PIT0.0</td>
      <td>ATL-0.5</td>
    </tr>
    <tr>
      <th>474</th>
      <td>2019</td>
      <td>LAA</td>
      <td>57</td>
      <td>28.8</td>
      <td>4.75</td>
      <td>162</td>
      <td>6251</td>
      <td>5542</td>
      <td>769</td>
      <td>1368</td>
      <td>...</td>
      <td>HOU0.0</td>
      <td>PHI0.4</td>
      <td>ARI1.1</td>
      <td>CLE1.6</td>
      <td>PIT-0.2</td>
      <td>MIL0.8</td>
      <td>LAA0.2</td>
      <td>TEX0.3</td>
      <td>CIN0.0</td>
      <td>MIL-0.6</td>
    </tr>
    <tr>
      <th>475</th>
      <td>2019</td>
      <td>LAD</td>
      <td>46</td>
      <td>27.9</td>
      <td>5.47</td>
      <td>162</td>
      <td>6282</td>
      <td>5493</td>
      <td>886</td>
      <td>1414</td>
      <td>...</td>
      <td>MIN-0.3</td>
      <td>MIN0.3</td>
      <td>CLE1.0</td>
      <td>TOR1.4</td>
      <td>MIN-0.3</td>
      <td>TBR0.5</td>
      <td>CHC0.2</td>
      <td>NYM0.1</td>
      <td>CHC0.0</td>
      <td>LAA-0.7</td>
    </tr>
    <tr>
      <th>476</th>
      <td>2019</td>
      <td>MIA</td>
      <td>50</td>
      <td>28.4</td>
      <td>3.80</td>
      <td>162</td>
      <td>6045</td>
      <td>5512</td>
      <td>615</td>
      <td>1326</td>
      <td>...</td>
      <td>CIN-0.5</td>
      <td>WSN0.2</td>
      <td>SFG0.8</td>
      <td>LAA0.5</td>
      <td>ARI-0.3</td>
      <td>BOS0.4</td>
      <td>CIN0.1</td>
      <td>ARI0.1</td>
      <td>WSN0.0</td>
      <td>WSN-0.7</td>
    </tr>
    <tr>
      <th>477</th>
      <td>2019</td>
      <td>MIL</td>
      <td>50</td>
      <td>28.9</td>
      <td>4.75</td>
      <td>162</td>
      <td>6309</td>
      <td>5542</td>
      <td>769</td>
      <td>1366</td>
      <td>...</td>
      <td>PHI-0.8</td>
      <td>CHW-0.1</td>
      <td>NYM0.8</td>
      <td>MIA0.4</td>
      <td>ATL-0.3</td>
      <td>SDP-0.2</td>
      <td>COL0.0</td>
      <td>STL0.0</td>
      <td>NYM0.0</td>
      <td>SFG-0.7</td>
    </tr>
    <tr>
      <th>478</th>
      <td>2019</td>
      <td>MIN</td>
      <td>50</td>
      <td>27.8</td>
      <td>5.80</td>
      <td>162</td>
      <td>6392</td>
      <td>5732</td>
      <td>939</td>
      <td>1547</td>
      <td>...</td>
      <td>MIL-0.8</td>
      <td>KCR-0.4</td>
      <td>SDP0.5</td>
      <td>NYY0.4</td>
      <td>SFG-0.5</td>
      <td>PHI-0.2</td>
      <td>BAL-0.1</td>
      <td>CHC-1.0</td>
      <td>SDP-0.1</td>
      <td>NYM-0.8</td>
    </tr>
    <tr>
      <th>479</th>
      <td>2019</td>
      <td>NYM</td>
      <td>53</td>
      <td>27.9</td>
      <td>4.88</td>
      <td>162</td>
      <td>6290</td>
      <td>5624</td>
      <td>791</td>
      <td>1445</td>
      <td>...</td>
      <td>WSN-1.0</td>
      <td>NYM-0.8</td>
      <td>STL0.5</td>
      <td>PIT0.1</td>
      <td>TOR-0.6</td>
      <td>CHW-0.8</td>
      <td>ATL-0.2</td>
      <td>CLE-1.6</td>
      <td>STL-0.1</td>
      <td>KCR-0.9</td>
    </tr>
    <tr>
      <th>480</th>
      <td>2019</td>
      <td>NYY</td>
      <td>54</td>
      <td>28.3</td>
      <td>5.82</td>
      <td>162</td>
      <td>6245</td>
      <td>5583</td>
      <td>943</td>
      <td>1493</td>
      <td>...</td>
      <td>CHW-1.1</td>
      <td>SDP-0.9</td>
      <td>SEA0.3</td>
      <td>KCR-0.1</td>
      <td>BOS-0.8</td>
      <td>SFG-0.9</td>
      <td>WSN-0.3</td>
      <td>PIT-1.7</td>
      <td>SFG-0.1</td>
      <td>TOR-0.9</td>
    </tr>
    <tr>
      <th>481</th>
      <td>2019</td>
      <td>OAK</td>
      <td>49</td>
      <td>27.8</td>
      <td>5.22</td>
      <td>162</td>
      <td>6269</td>
      <td>5561</td>
      <td>845</td>
      <td>1384</td>
      <td>...</td>
      <td>BOS-1.3</td>
      <td>CLE-1.1</td>
      <td>MIA0.0</td>
      <td>TEX-0.4</td>
      <td>CLE-0.9</td>
      <td>CLE-0.9</td>
      <td>SDP-0.6</td>
      <td>KCR-2.3</td>
      <td>PHI-0.1</td>
      <td>CIN-1.0</td>
    </tr>
    <tr>
      <th>482</th>
      <td>2019</td>
      <td>PHI</td>
      <td>56</td>
      <td>27.7</td>
      <td>4.78</td>
      <td>162</td>
      <td>6261</td>
      <td>5571</td>
      <td>774</td>
      <td>1369</td>
      <td>...</td>
      <td>MIA-1.4</td>
      <td>SEA-1.3</td>
      <td>BAL-0.3</td>
      <td>NYM-0.5</td>
      <td>KCR-1.0</td>
      <td>CIN-1.4</td>
      <td>STL-0.8</td>
      <td>SFG-2.3</td>
      <td>ARI-0.1</td>
      <td>CHC-1.0</td>
    </tr>
    <tr>
      <th>483</th>
      <td>2019</td>
      <td>PIT</td>
      <td>54</td>
      <td>27.5</td>
      <td>4.68</td>
      <td>162</td>
      <td>6228</td>
      <td>5657</td>
      <td>758</td>
      <td>1497</td>
      <td>...</td>
      <td>SFG-1.4</td>
      <td>OAK-1.4</td>
      <td>LAA-0.3</td>
      <td>PHI-0.5</td>
      <td>LAA-1.3</td>
      <td>KCR-1.5</td>
      <td>SEA-0.8</td>
      <td>SDP-2.4</td>
      <td>COL-0.2</td>
      <td>BOS-1.2</td>
    </tr>
    <tr>
      <th>484</th>
      <td>2019</td>
      <td>SDP</td>
      <td>54</td>
      <td>26.2</td>
      <td>4.21</td>
      <td>162</td>
      <td>6019</td>
      <td>5391</td>
      <td>682</td>
      <td>1281</td>
      <td>...</td>
      <td>BAL-1.7</td>
      <td>SFG-1.4</td>
      <td>TBR-0.5</td>
      <td>ATL-0.7</td>
      <td>SEA-1.4</td>
      <td>NYM-1.6</td>
      <td>TEX-0.9</td>
      <td>CIN-3.1</td>
      <td>SEA-0.2</td>
      <td>SDP-1.2</td>
    </tr>
    <tr>
      <th>485</th>
      <td>2019</td>
      <td>SEA</td>
      <td>67</td>
      <td>27.8</td>
      <td>4.68</td>
      <td>162</td>
      <td>6199</td>
      <td>5500</td>
      <td>758</td>
      <td>1305</td>
      <td>...</td>
      <td>TEX-1.7</td>
      <td>COL-1.6</td>
      <td>TOR-0.6</td>
      <td>BAL-0.8</td>
      <td>SDP-1.6</td>
      <td>CHC-1.7</td>
      <td>SFG-0.9</td>
      <td>TOR-4.8</td>
      <td>NYY-0.3</td>
      <td>PHI-1.3</td>
    </tr>
    <tr>
      <th>486</th>
      <td>2019</td>
      <td>SFG</td>
      <td>64</td>
      <td>29.9</td>
      <td>4.19</td>
      <td>162</td>
      <td>6170</td>
      <td>5579</td>
      <td>678</td>
      <td>1332</td>
      <td>...</td>
      <td>TOR-1.7</td>
      <td>CHC-1.9</td>
      <td>MIL-0.6</td>
      <td>CIN-0.8</td>
      <td>CHW-1.7</td>
      <td>DET-1.9</td>
      <td>MIA-1.0</td>
      <td>BAL-4.8</td>
      <td>TOR-0.8</td>
      <td>BAL-1.4</td>
    </tr>
    <tr>
      <th>487</th>
      <td>2019</td>
      <td>STL</td>
      <td>43</td>
      <td>28.8</td>
      <td>4.72</td>
      <td>162</td>
      <td>6167</td>
      <td>5449</td>
      <td>764</td>
      <td>1336</td>
      <td>...</td>
      <td>DET-2.0</td>
      <td>TEX-2.1</td>
      <td>KCR-1.2</td>
      <td>SFG-1.0</td>
      <td>CIN-1.8</td>
      <td>TOR-1.9</td>
      <td>DET-2.1</td>
      <td>SEA-4.9</td>
      <td>BAL-1.1</td>
      <td>CHW-1.6</td>
    </tr>
    <tr>
      <th>488</th>
      <td>2019</td>
      <td>TBR</td>
      <td>57</td>
      <td>27.2</td>
      <td>4.75</td>
      <td>162</td>
      <td>6285</td>
      <td>5628</td>
      <td>769</td>
      <td>1427</td>
      <td>...</td>
      <td>LAA-2.3</td>
      <td>BOS-2.1</td>
      <td>PHI-1.3</td>
      <td>SEA-1.0</td>
      <td>DET-2.2</td>
      <td>BAL-2.0</td>
      <td>ARI-2.1</td>
      <td>COL-5.8</td>
      <td>CLE-1.4</td>
      <td>DET-1.6</td>
    </tr>
    <tr>
      <th>489</th>
      <td>2019</td>
      <td>TEX</td>
      <td>53</td>
      <td>28.8</td>
      <td>5.00</td>
      <td>162</td>
      <td>6204</td>
      <td>5540</td>
      <td>810</td>
      <td>1374</td>
      <td>...</td>
      <td>COL-2.3</td>
      <td>MIA-2.3</td>
      <td>TEX-1.9</td>
      <td>WSN-1.4</td>
      <td>BAL-2.7</td>
      <td>SEA-2.7</td>
      <td>TOR-2.3</td>
      <td>DET-6.2</td>
      <td>OAK-1.8</td>
      <td>PIT-2.1</td>
    </tr>
    <tr>
      <th>490</th>
      <td>2019</td>
      <td>TOR</td>
      <td>61</td>
      <td>25.9</td>
      <td>4.48</td>
      <td>162</td>
      <td>6091</td>
      <td>5493</td>
      <td>726</td>
      <td>1299</td>
      <td>...</td>
      <td>SDP-2.4</td>
      <td>CIN-2.4</td>
      <td>DET-2.1</td>
      <td>DET-2.0</td>
      <td>COL-2.8</td>
      <td>COL-3.0</td>
      <td>PIT-2.7</td>
      <td>CHW-6.3</td>
      <td>CHW-2.5</td>
      <td>COL-2.8</td>
    </tr>
    <tr>
      <th>491</th>
      <td>2019</td>
      <td>WSN</td>
      <td>50</td>
      <td>28.8</td>
      <td>5.39</td>
      <td>162</td>
      <td>6267</td>
      <td>5512</td>
      <td>873</td>
      <td>1460</td>
      <td>...</td>
      <td>KCR-4.0</td>
      <td>DET-2.9</td>
      <td>PIT-2.5</td>
      <td>MIL-2.2</td>
      <td>MIA-3.3</td>
      <td>MIA-3.6</td>
      <td>CHW-3.8</td>
      <td>MIA-7.9</td>
      <td>DET-2.6</td>
      <td>MIA-3.0</td>
    </tr>
    <tr>
      <th>492</th>
      <td>2019</td>
      <td>LgAvg</td>
      <td>47</td>
      <td>27.9</td>
      <td>4.83</td>
      <td>162</td>
      <td>6217</td>
      <td>5555</td>
      <td>782</td>
      <td>1401</td>
      <td>...</td>
      <td>-0.3</td>
      <td>-0.1</td>
      <td>0.9</td>
      <td>1.0</td>
      <td>-0.4</td>
      <td>0.1</td>
      <td>0.2</td>
      <td>-0.1</td>
      <td>-0.1</td>
      <td>-0.8</td>
    </tr>
    <tr>
      <th>493</th>
      <td>2019</td>
      <td>Tm</td>
      <td>#Bat</td>
      <td>BatAge</td>
      <td>R/G</td>
      <td>G</td>
      <td>PA</td>
      <td>AB</td>
      <td>R</td>
      <td>H</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>494</th>
      <td>2019</td>
      <td>NaN</td>
      <td>1410</td>
      <td>27.9</td>
      <td>4.83</td>
      <td>4858</td>
      <td>186517</td>
      <td>166651</td>
      <td>23467</td>
      <td>42039</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>495 rows × 84 columns</p>
</div>



<h3>1.4 End Goal Product</h3>
<p>
    Before we dive into our analysis, research, and model creation, it's important to lay out our end goals so we
    can stay focused throughout the process as well as the format of our eventual predicted win totals.  Our End
    Product should contain:
    <ul>
        <li>Win Totals For Each Team</li>
        <li>Projected Playoff Teams</li>
    </ul>
    <br>
    The output will be a <b>DataFrame</b> with the columns <b>Team, League, Division, Projected Wins and
    Playoffs?</b>  The projected 6 Division Winners, 2 First Wild Card, and 2 Second Wild Card
    teams will be outputted as well.  Since we are not involving the teams schedules, any potential ties will be
    decided by alphabetical order, although ties are unlikely as wins will be predicted to more than a few decimals
    of precision.
</p>

<h2><center>2 What Statistics Contribute to More Wins?</center></h2>

<h3>2.1 Choosing a Starting Point</h3>
<p>
    With so many new statistics and vast amounts of data in modern-day baseball, it can seem quite daunting to try
    and analyze every possible statistic and its relationship to number of wins and then somehow piece it together.
    The best models are constantly being reworked, tweaked, and updated in order to be as accurate as possible, but
    they don't simply go from raw data to a predictive model.  They all start from the ground up, finding statistics
    that have strong correlations with wins first and then moving up from there.
    <br><br>
    As an avid baseball fan who already has experience with some basic forms of sabermetrics (such as in fantasy
    baseball), I have a general idea of where I want to start looking for correlations, but if you are not as big of
    a baseball fan, then there are many resources on the internet that can provide you some good starting points.
    As stated before, we will try not to incorporate too many statistics into the end product for the purposes of
    this tutorial as that would take an extensive amount of time and likely lose the reader in the process.
</p>


<h3>2.2 Example of Relationship Analyzation: Batting Average!</h3>
<p>
    One of the most well-known baseball statistics is batting average.  Batting Average for an entire team is similar
    to how it is calculated for an individual player: <b>.AVG = Teams' Total # of Hits / Teams' Total # of At-
    Bats</b>.
    <br><br>
    Baseball Reference provides these pieces of information for every team in every season.  In addition, they
    already have the batting averages calculated in its own column, so we do not have to calculate it ourselves.
</p>
<h4><center>Analyzing for a Correlation</center></h4>
<p>
    To analyze a potential correlation between batting average and regular season wins, we need to plot <b>wins vs.
    team batting average</b>.  To do this, we need to draw from our season wins dataframe and our master stats
    dataframe in order to create a new 2 column dataframe of: a team from a given season's batting average and that
    same team in that given season's total regular season wins.  Then, we plot all 15 seasons of this data and fit
    a linear regression model to it.
</p>


```python
# Creating the dataframe to plot
df_winsbavg = pd.DataFrame()

build_wincol = []

# Need to reference each row of the total wins table to append each column value in order to match up with batting avg
for row in df_twt.values:
    for col_ind in range(1, 31):
        build_wincol.append(float(row[col_ind]))

# Need to remove any rows that do not contain just a batting average in a copy of df_master
copy_dfm = df_master[df_master.BA.str[0] == "."]
copy_dfm = copy_dfm[copy_dfm["Tm"] != "LgAvg"]
copy_dfm = copy_dfm[copy_dfm["Tm"] != "Tm"]
copy_dfm = copy_dfm[copy_dfm["Tm"] != "BA"]
copy_dfm = copy_dfm[copy_dfm["Tm"] != np.nan]
copy_dfm = copy_dfm[pd.isnull(copy_dfm["Tm"]) == False]
copy_dfm["BA"] = copy_dfm["BA"].astype(float)
copy_dfm = copy_dfm.reset_index(drop = True)

# Now we can add build_wincol as the wins column with all indicies matching up for their respective batting avg data
df_winsbavg["Batting Avg"] = copy_dfm["BA"]
df_winsbavg["Wins"] = build_wincol

# Our dataframe is now ready to plot:
df_winsbavg
```




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
      <th>Batting Avg</th>
      <th>Wins</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.256</td>
      <td>77.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.265</td>
      <td>90.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.269</td>
      <td>74.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.281</td>
      <td>95.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.270</td>
      <td>79.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.262</td>
      <td>99.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.261</td>
      <td>73.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.271</td>
      <td>93.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.267</td>
      <td>67.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.272</td>
      <td>71.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.272</td>
      <td>89.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.256</td>
      <td>56.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.263</td>
      <td>95.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.270</td>
      <td>71.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.253</td>
      <td>83.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.259</td>
      <td>81.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.259</td>
      <td>83.0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.258</td>
      <td>83.0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.276</td>
      <td>95.0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0.262</td>
      <td>88.0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.270</td>
      <td>88.0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0.259</td>
      <td>67.0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>0.257</td>
      <td>82.0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>0.256</td>
      <td>75.0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>0.261</td>
      <td>69.0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>0.270</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>0.274</td>
      <td>67.0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>0.267</td>
      <td>79.0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>0.265</td>
      <td>80.0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>0.252</td>
      <td>81.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>420</th>
      <td>0.252</td>
      <td>85.0</td>
    </tr>
    <tr>
      <th>421</th>
      <td>0.258</td>
      <td>97.0</td>
    </tr>
    <tr>
      <th>422</th>
      <td>0.246</td>
      <td>54.0</td>
    </tr>
    <tr>
      <th>423</th>
      <td>0.269</td>
      <td>84.0</td>
    </tr>
    <tr>
      <th>424</th>
      <td>0.252</td>
      <td>84.0</td>
    </tr>
    <tr>
      <th>425</th>
      <td>0.261</td>
      <td>72.0</td>
    </tr>
    <tr>
      <th>426</th>
      <td>0.244</td>
      <td>75.0</td>
    </tr>
    <tr>
      <th>427</th>
      <td>0.250</td>
      <td>93.0</td>
    </tr>
    <tr>
      <th>428</th>
      <td>0.265</td>
      <td>71.0</td>
    </tr>
    <tr>
      <th>429</th>
      <td>0.240</td>
      <td>47.0</td>
    </tr>
    <tr>
      <th>430</th>
      <td>0.274</td>
      <td>107.0</td>
    </tr>
    <tr>
      <th>431</th>
      <td>0.247</td>
      <td>59.0</td>
    </tr>
    <tr>
      <th>432</th>
      <td>0.247</td>
      <td>72.0</td>
    </tr>
    <tr>
      <th>433</th>
      <td>0.257</td>
      <td>106.0</td>
    </tr>
    <tr>
      <th>434</th>
      <td>0.241</td>
      <td>57.0</td>
    </tr>
    <tr>
      <th>435</th>
      <td>0.246</td>
      <td>89.0</td>
    </tr>
    <tr>
      <th>436</th>
      <td>0.270</td>
      <td>101.0</td>
    </tr>
    <tr>
      <th>437</th>
      <td>0.257</td>
      <td>86.0</td>
    </tr>
    <tr>
      <th>438</th>
      <td>0.267</td>
      <td>103.0</td>
    </tr>
    <tr>
      <th>439</th>
      <td>0.249</td>
      <td>97.0</td>
    </tr>
    <tr>
      <th>440</th>
      <td>0.246</td>
      <td>81.0</td>
    </tr>
    <tr>
      <th>441</th>
      <td>0.265</td>
      <td>69.0</td>
    </tr>
    <tr>
      <th>442</th>
      <td>0.238</td>
      <td>70.0</td>
    </tr>
    <tr>
      <th>443</th>
      <td>0.237</td>
      <td>77.0</td>
    </tr>
    <tr>
      <th>444</th>
      <td>0.239</td>
      <td>68.0</td>
    </tr>
    <tr>
      <th>445</th>
      <td>0.245</td>
      <td>91.0</td>
    </tr>
    <tr>
      <th>446</th>
      <td>0.254</td>
      <td>96.0</td>
    </tr>
    <tr>
      <th>447</th>
      <td>0.248</td>
      <td>78.0</td>
    </tr>
    <tr>
      <th>448</th>
      <td>0.236</td>
      <td>67.0</td>
    </tr>
    <tr>
      <th>449</th>
      <td>0.265</td>
      <td>93.0</td>
    </tr>
  </tbody>
</table>
<p>450 rows × 2 columns</p>
</div>




```python
# Rename copy_dfm to df_research for clarity
df_research = copy_dfm

# Now we want to plot this and fit a linear regression to the plot using Seaborn and sklearn's linear_model
lreg_winsbavg = lmod.LinearRegression()
lreg_X = [[x] for x in df_winsbavg["Batting Avg"].values]
lreg_Y = [[y] for y in df_winsbavg["Wins"].values]
lreg_fit = lreg_winsbavg.fit(lreg_X, lreg_Y)

regplot_winsbavg = sea.regplot(x = "Batting Avg", y = "Wins", data = df_winsbavg)
regplot_winsbavg.set(xlabel = "Team Batting Avg", ylabel = "Total Regular Season Wins",
                     title = "Total Regular Season Wins vs. Team Batting Avg")
regplot_winsbavg
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fb56609ff60>




![png](tutorial-jasonschneider_files/tutorial-jasonschneider_14_1.png)



```python
# From the graph above, it seems like a higher team batting average tends to contribute to more regular season wins.
# Using the linear regression we can create a linear equation in the form y = mx + b to predict how many regular
# season wins are expected based off of JUST a hypothetical whole season team batting average.
slope0 = lreg_winsbavg.coef_[0][0]
intercept0 = lreg_winsbavg.intercept_[0]

print("# of Regular Season Wins = " + str(slope0) + " * <Batting Avg> + " + str(intercept0))
print("Pearson Correlation: " + str(pr(df_winsbavg["Batting Avg"], df_winsbavg["Wins"])[0]))

# You've just completed your first basic statistic analyzation!
```

    # of Regular Season Wins = 300.3002325179685 * <Batting Avg> + 3.6433407808888205
    Pearson Correlation: 0.313334368854188


<h4><center>Results and Where to Go From Here</center></h4>
<p>
    The linear model that you just created is technically a predictive model by itself already, however, it is not
    very accurate as it is based around just one statistic, claiming there is a perfect correlation between a team's
    batting average in a season and their total regular season wins in that season.  We <i>have</i> found that there
    is a <b>moderate</b> correlation between a team's batting average and their season win total, with a Pearson
    Correlation of about 0.313.  This makes sense because a team that gets more hits is getting more runners
    on base, thus increasing the chances of scoring or increasing the number of runs scored every game.
    <br><br>
    From here, it is up to the data scientist to research more trends and relationships on their own in order to
    add the number of factors involved in projected regular season win totals for each MLB team.  This portion of
    the process for large-scale, highly-funded models is one of the most, if not the most, time-consuming parts of
    the predictive model creation process, as this is where the model is fine-tuned to account for the most up to
    date ideas in sabermetrical analysis of baseball.
    <br><br>
    In the next few sections, we will be performing the same steps to analyze how the following statistics contribute
    to total regular season wins: <b>Runs Scored</b> (another batting statistic), <b>WHIP</b> (a pitching statistic),
    and <b><i>WAR</i></b> (advanced statistic for both pitchers and field players).  <i>Note: There will be less
    explanation along the way of steps we already explained in the example.</i>
    
</p>

<h3>2.3 Relationship: Runs Scored and Wins</h3>
<p>
    Next, I am going to explore the relationship between runs scored and wins.  Do we expect there to be a
    correlation?  Most likely, yes, as teams that score more runs during a whole season also score more runs during
    every game on average than teams that score less runs during a whole season.
</p>


```python
# Rename df_winsbavg to df_factors as this will now maintain important data on factors involved in our eventual
# complete predictive model
df_factors = df_winsbavg

# Add years, teams, and runs scored to our factors table
df_factors.insert(0, "Year", df_research["Year"].astype(int))
df_factors.insert(1, "Team", df_research["Tm"])
df_factors["Runs Scored"] = df_research["R_b"].astype(float)
df_factors
```




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
      <th>Year</th>
      <th>Team</th>
      <th>Batting Avg</th>
      <th>Wins</th>
      <th>Runs Scored</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2005</td>
      <td>ARI</td>
      <td>0.256</td>
      <td>77.0</td>
      <td>696.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2005</td>
      <td>ATL</td>
      <td>0.265</td>
      <td>90.0</td>
      <td>769.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2005</td>
      <td>BAL</td>
      <td>0.269</td>
      <td>74.0</td>
      <td>729.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2005</td>
      <td>BOS</td>
      <td>0.281</td>
      <td>95.0</td>
      <td>910.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2005</td>
      <td>CHC</td>
      <td>0.270</td>
      <td>79.0</td>
      <td>703.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2005</td>
      <td>CHW</td>
      <td>0.262</td>
      <td>99.0</td>
      <td>741.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2005</td>
      <td>CIN</td>
      <td>0.261</td>
      <td>73.0</td>
      <td>820.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2005</td>
      <td>CLE</td>
      <td>0.271</td>
      <td>93.0</td>
      <td>790.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2005</td>
      <td>COL</td>
      <td>0.267</td>
      <td>67.0</td>
      <td>740.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2005</td>
      <td>DET</td>
      <td>0.272</td>
      <td>71.0</td>
      <td>723.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2005</td>
      <td>MIA</td>
      <td>0.272</td>
      <td>89.0</td>
      <td>717.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2005</td>
      <td>HOU</td>
      <td>0.256</td>
      <td>56.0</td>
      <td>693.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2005</td>
      <td>KCR</td>
      <td>0.263</td>
      <td>95.0</td>
      <td>701.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2005</td>
      <td>LAA</td>
      <td>0.270</td>
      <td>71.0</td>
      <td>761.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2005</td>
      <td>LAD</td>
      <td>0.253</td>
      <td>83.0</td>
      <td>685.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2005</td>
      <td>MIL</td>
      <td>0.259</td>
      <td>81.0</td>
      <td>726.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2005</td>
      <td>MIN</td>
      <td>0.259</td>
      <td>83.0</td>
      <td>688.0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2005</td>
      <td>NYM</td>
      <td>0.258</td>
      <td>83.0</td>
      <td>722.0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2005</td>
      <td>NYY</td>
      <td>0.276</td>
      <td>95.0</td>
      <td>886.0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2005</td>
      <td>OAK</td>
      <td>0.262</td>
      <td>88.0</td>
      <td>772.0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2005</td>
      <td>PHI</td>
      <td>0.270</td>
      <td>88.0</td>
      <td>807.0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2005</td>
      <td>PIT</td>
      <td>0.259</td>
      <td>67.0</td>
      <td>680.0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>2005</td>
      <td>SDP</td>
      <td>0.257</td>
      <td>82.0</td>
      <td>684.0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>2005</td>
      <td>SEA</td>
      <td>0.256</td>
      <td>75.0</td>
      <td>699.0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>2005</td>
      <td>SFG</td>
      <td>0.261</td>
      <td>69.0</td>
      <td>649.0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>2005</td>
      <td>STL</td>
      <td>0.270</td>
      <td>100.0</td>
      <td>805.0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>2005</td>
      <td>TBR</td>
      <td>0.274</td>
      <td>67.0</td>
      <td>750.0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>2005</td>
      <td>TEX</td>
      <td>0.267</td>
      <td>79.0</td>
      <td>865.0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>2005</td>
      <td>TOR</td>
      <td>0.265</td>
      <td>80.0</td>
      <td>775.0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>2005</td>
      <td>WSN</td>
      <td>0.252</td>
      <td>81.0</td>
      <td>639.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>420</th>
      <td>2019</td>
      <td>ARI</td>
      <td>0.252</td>
      <td>85.0</td>
      <td>813.0</td>
    </tr>
    <tr>
      <th>421</th>
      <td>2019</td>
      <td>ATL</td>
      <td>0.258</td>
      <td>97.0</td>
      <td>855.0</td>
    </tr>
    <tr>
      <th>422</th>
      <td>2019</td>
      <td>BAL</td>
      <td>0.246</td>
      <td>54.0</td>
      <td>729.0</td>
    </tr>
    <tr>
      <th>423</th>
      <td>2019</td>
      <td>BOS</td>
      <td>0.269</td>
      <td>84.0</td>
      <td>901.0</td>
    </tr>
    <tr>
      <th>424</th>
      <td>2019</td>
      <td>CHC</td>
      <td>0.252</td>
      <td>84.0</td>
      <td>814.0</td>
    </tr>
    <tr>
      <th>425</th>
      <td>2019</td>
      <td>CHW</td>
      <td>0.261</td>
      <td>72.0</td>
      <td>708.0</td>
    </tr>
    <tr>
      <th>426</th>
      <td>2019</td>
      <td>CIN</td>
      <td>0.244</td>
      <td>75.0</td>
      <td>701.0</td>
    </tr>
    <tr>
      <th>427</th>
      <td>2019</td>
      <td>CLE</td>
      <td>0.250</td>
      <td>93.0</td>
      <td>769.0</td>
    </tr>
    <tr>
      <th>428</th>
      <td>2019</td>
      <td>COL</td>
      <td>0.265</td>
      <td>71.0</td>
      <td>835.0</td>
    </tr>
    <tr>
      <th>429</th>
      <td>2019</td>
      <td>DET</td>
      <td>0.240</td>
      <td>47.0</td>
      <td>582.0</td>
    </tr>
    <tr>
      <th>430</th>
      <td>2019</td>
      <td>HOU</td>
      <td>0.274</td>
      <td>107.0</td>
      <td>920.0</td>
    </tr>
    <tr>
      <th>431</th>
      <td>2019</td>
      <td>KCR</td>
      <td>0.247</td>
      <td>59.0</td>
      <td>691.0</td>
    </tr>
    <tr>
      <th>432</th>
      <td>2019</td>
      <td>LAA</td>
      <td>0.247</td>
      <td>72.0</td>
      <td>769.0</td>
    </tr>
    <tr>
      <th>433</th>
      <td>2019</td>
      <td>LAD</td>
      <td>0.257</td>
      <td>106.0</td>
      <td>886.0</td>
    </tr>
    <tr>
      <th>434</th>
      <td>2019</td>
      <td>MIA</td>
      <td>0.241</td>
      <td>57.0</td>
      <td>615.0</td>
    </tr>
    <tr>
      <th>435</th>
      <td>2019</td>
      <td>MIL</td>
      <td>0.246</td>
      <td>89.0</td>
      <td>769.0</td>
    </tr>
    <tr>
      <th>436</th>
      <td>2019</td>
      <td>MIN</td>
      <td>0.270</td>
      <td>101.0</td>
      <td>939.0</td>
    </tr>
    <tr>
      <th>437</th>
      <td>2019</td>
      <td>NYM</td>
      <td>0.257</td>
      <td>86.0</td>
      <td>791.0</td>
    </tr>
    <tr>
      <th>438</th>
      <td>2019</td>
      <td>NYY</td>
      <td>0.267</td>
      <td>103.0</td>
      <td>943.0</td>
    </tr>
    <tr>
      <th>439</th>
      <td>2019</td>
      <td>OAK</td>
      <td>0.249</td>
      <td>97.0</td>
      <td>845.0</td>
    </tr>
    <tr>
      <th>440</th>
      <td>2019</td>
      <td>PHI</td>
      <td>0.246</td>
      <td>81.0</td>
      <td>774.0</td>
    </tr>
    <tr>
      <th>441</th>
      <td>2019</td>
      <td>PIT</td>
      <td>0.265</td>
      <td>69.0</td>
      <td>758.0</td>
    </tr>
    <tr>
      <th>442</th>
      <td>2019</td>
      <td>SDP</td>
      <td>0.238</td>
      <td>70.0</td>
      <td>682.0</td>
    </tr>
    <tr>
      <th>443</th>
      <td>2019</td>
      <td>SEA</td>
      <td>0.237</td>
      <td>77.0</td>
      <td>758.0</td>
    </tr>
    <tr>
      <th>444</th>
      <td>2019</td>
      <td>SFG</td>
      <td>0.239</td>
      <td>68.0</td>
      <td>678.0</td>
    </tr>
    <tr>
      <th>445</th>
      <td>2019</td>
      <td>STL</td>
      <td>0.245</td>
      <td>91.0</td>
      <td>764.0</td>
    </tr>
    <tr>
      <th>446</th>
      <td>2019</td>
      <td>TBR</td>
      <td>0.254</td>
      <td>96.0</td>
      <td>769.0</td>
    </tr>
    <tr>
      <th>447</th>
      <td>2019</td>
      <td>TEX</td>
      <td>0.248</td>
      <td>78.0</td>
      <td>810.0</td>
    </tr>
    <tr>
      <th>448</th>
      <td>2019</td>
      <td>TOR</td>
      <td>0.236</td>
      <td>67.0</td>
      <td>726.0</td>
    </tr>
    <tr>
      <th>449</th>
      <td>2019</td>
      <td>WSN</td>
      <td>0.265</td>
      <td>93.0</td>
      <td>873.0</td>
    </tr>
  </tbody>
</table>
<p>450 rows × 5 columns</p>
</div>




```python
# Plot, Fit Linear Regression, Create equation, and find Pearson Correlation similar to example
lreg_winsruns = lmod.LinearRegression()
winsruns_X = [[x] for x in df_factors["Runs Scored"].values]
winsruns_Y = [[y] for y in df_factors["Wins"].values]
winsruns_fit = lreg_winsruns.fit(winsruns_X, winsruns_Y)

regplot_winsruns = sea.regplot(x = "Runs Scored", y = "Wins", data = df_factors)
regplot_winsruns.set(xlabel = "Team Runs Scored", ylabel = "Total Regular Season Wins",
                     title = "Total Regular Season Wins vs. Team Runs Scored")
regplot_winsruns
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fb5641d44a8>




![png](tutorial-jasonschneider_files/tutorial-jasonschneider_19_1.png)



```python
# The plot above shows that there is an even stronger correlation between Team Runs Scored in a Season and Total
# Regular Season Wins for that season.  The equation for the regression in y = mx + b format is:
slope1 = lreg_winsruns.coef_[0][0]
intercept1 = lreg_winsruns.intercept_[0]

print("# of Regular Season Wins = " + str(slope1) + " * <Runs Scored> + " + str(intercept1))
print("Pearson Correlation: " + str(pr(df_factors["Runs Scored"], df_factors["Wins"])[0]))
```

    # of Regular Season Wins = 0.07568260885548721 * <Runs Scored> + 25.911759468227686
    Pearson Correlation: 0.5189365795325355


<h4><center>Analysis Results</center></h4>
<p>
    As expected, there is a clear correlation between team runs scored in a season and wins in a season.  As stated
    before, a team that scores more runs is stastically more likely to win more games than a team that scores less
    runs as scoring runs is the only way to raise your teams' score in baseball, a sport where the objective is to
    score more runs than the other team.
    <br><br>
    The equation for the linear regression is given in the output above and the Pearson Correlation is about 0.519,
    which is a solid, moderate positive correlation.
</p>

<h3>2.4 Relationship: WHIP and Wins</h3>
<p>
    <b>WHIP</b> is a pitching statistic that stands for <b>Walks plus Hits per Innings Pitched</b>, or in other
    terms, it's the number of baserunners a team allows to safely reach base per innning.  A WHIP of 0.0 means that
    a team never allowed a baserunner to safely reach base while a WHIP of 2.0 means that a team allows an average
    of 2.0 baserunners to safely reach base per inning.  A team with a lower season WHIP means its pitchers performed
    well that season while a team with a higher season whip had pitchers that performed more poorly.  In general,
    anything less than 1.0 is elite, while most players' and teams' WHIP will tend to be somewhere between 1.2 and
    1.8.  We are likely to see that WHIP is negatively correlated with Wins, as lower numbers means better.
</p>


```python
# Add WHIP to factors table
df_factors["WHIP"] = df_research["WHIP"].astype(float)

# Plot, Fit Linear Regression, Create equation, and find Pearson Correlation similar to example
lreg_winswhip = lmod.LinearRegression()
winswhip_X = [[x] for x in df_factors["WHIP"].values]
winswhip_Y = [[y] for y in df_factors["Wins"].values]
winswhip_fit = lreg_winswhip.fit(winswhip_X, winswhip_Y)

regplot_winswhip = sea.regplot(x = "WHIP", y = "Wins", data = df_factors)
regplot_winswhip.set(xlabel = "Team WHIP", ylabel = "Total Regular Season Wins",
                     title = "Total Regular Season Wins vs. Team WHIP")
regplot_winswhip
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fb565103358>




![png](tutorial-jasonschneider_files/tutorial-jasonschneider_23_1.png)



```python
# The plot above shows that there is an evident negative correlation between Team WHIP and Total Regular Season Wins.
# The equation for the regression in y = mx + b format is:
slope2 = lreg_winswhip.coef_[0][0]
intercept2 = lreg_winswhip.intercept_[0]

print("# of Regular Season Wins = " + str(slope2) + " * <WHIP> + " + str(intercept2))
print("Pearson Correlation: " + str(pr(df_factors["WHIP"], df_factors["Wins"])[0]))
```

    # of Regular Season Wins = -73.11937611201859 * <WHIP> + 179.04576207047504
    Pearson Correlation: -0.5865951742238298


<h4><center>Analysis Results</center></h4>
<p>
    As expected, there is a negative correlation between Team WHIP for a season and wins in a season.  As stated
    before, a team that lets less runners on base per innings is going to end up giving up less runs as baserunners
    are required to score runs.
    <br><br>
    The equation for the linear regression is given in the output above and the Pearson Correlation is about -0.587,
    which is a solid, moderate negative correlation.
</p>

<h3>2.5 Relationship: WAR and Wins</h3>
<p>
    <b>WAR</b>, short for <b>Wins Above Replacement</b>, measures the "number of wins" a player's presence in the
    lineup for a whole 162-game season adds or subtracts from the team's overall record if the average league player
    were to play instead.  A positive WAR means the player generally helps the team while a negative war means the
    player is costing their team wins (causing them to lose more).  Calculating WAR involves several calculations and
    involves many different statistics, but we won't need to do that as Baseball Reference has done the calculations
    for us.
    <br><br>
    The data for Team WAR is an average of the players' WARs for a given season, which means we can use Team WAR to
    take a quick glance at the teams with the most "win-helpful" players.
</p>


```python
# Re-format/Sort and then add WAR to factors table
build_warscol = pd.Series([])
switch = {
    "ARI" : 0,
    "ATL" : 1,
    "BAL" : 2,
    "BOS" : 3,
    "CHC" : 4,
    "CHW" : 5,
    "CIN" : 6,
    "CLE" : 7,
    "COL" : 8,
    "DET" : 9,
    "MIA" : 10,
    "HOU" : 11,
    "KCR" : 12,
    "LAA" : 13,
    "LAD" : 14,
    "MIL" : 15,
    "MIN" : 16,
    "NYM" : 17,
    "NYY" : 18,
    "OAK" : 19,
    "PHI" : 20,
    "PIT" : 21,
    "SDP" : 22,
    "SEA" : 23,
    "SFG" : 24,
    "STL" : 25,
    "TBR" : 26,
    "TEX" : 27,
    "TOR" : 28,
    "WSN" : 29,
    "FLA" : -1
}
for yr in range(2005, 2020):
    wars = df_research[df_research["Year"] == yr]
    wars = wars["Total"]
    wars = wars.replace("TBD", "TBR")
    wars = wars.replace("FLA", "MIA")
    build_local = []
    
    for war in wars.astype(str):
        team_code = war[0] + war[1] + war[2]
        local_index = switch.get(team_code)
        if local_index is not None:
            build_local.insert(local_index, team_code)
    if build_warscol is None:
        build_warscol = build_local
    else:
        build_warscol = build_warscol.append(pd.Series(build_local))
build_warscol = build_warscol.reset_index()
```


```python
# Then, you would average out all of you models against the average win amount for each team over the past 15 seasons
# and then find the average prediction of all the mini models.
```
