# **Statistics from scraping**

## **Finding which websites to scrape**

### **Method**

Loaded dummy data set from umass extension and searched bing for urls

Then checked for how the number of links from each site that just searched by */common_name* or */scientific_name*

### **Top 5 Results**

### Sci score results

**plants.ces.ncsu.edu: 170 -> 0.33** <br>
rhs.org.uk: 38 -> 0.08 <br>
gardenersworld.com: 25 -> 0.04 <br>
hort.extension.wisc.edu: 34 -> 0.03 <br>
americanmeadows.com: 36 -> 0.03 <br>

### Common name score results

extension.umass.edu: 22 -> 0.73 (example ds) <br>
**britannica.com: 23 -> 0.61** <br>
**epicgardening.com: 40 -> 0.23** <br>
gardenerspath.com: 45 -> 0.11 <br>
provenwinners.com: 75 -> 0.11 <br>

### **Plan**

1. Search plants.ces.ncsu.edu and wikipedia with search
2. Find the description tag and common names
3. Use the common names to find more descriptions from britannica and epicgardening.com

### **Scraping Stats (Scientific Names)**

*Overall 15501 unique plants*
#### **Wikipedia**

<table>
    <tr>
        <th>Stat</th>
        <th>Number</th>
        <th>% Total</th>
    </tr>
    <tr>
        <td>Page Exists</td>
        <td>10212</td>
        <td>66%</td>
    </tr>
    <tr>
        <td>Common Names</td>
        <td>8217</td>
        <td>53%</td>
    </tr>
    <tr>
        <td>Description</td>
        <td>3960</td>
        <td>26%</td>
    </tr>
</table>

*Covers 8736 pages that NCSU does not*

While the number of descriptions is low can be augmented by finding non taged description and using common names for further scraping.

#### **NCSU**

<table>
    <tr>
        <th>Stat</th>
        <th>Number</th>
        <th>% Total</th>
    </tr>
    <tr>
        <td>Page Exists</td>
        <td>1539</td>
        <td>10%</td>
    </tr>
    <tr>
        <td>Common Names</td>
        <td>1537</td>
        <td>10%</td>
    </tr>
    <tr>
        <td>Description</td>
        <td>1539</td>
        <td>10%</td>
    </tr>
</table>

*Covers 63 pages that Wiki does not*

Number of pages is low but offers good description coverage as well as common names

#### **Combined**

<table>
    <tr>
        <th>Stat</th>
        <th>Number</th>
        <th>% Total</th>
    </tr>
    <tr>
        <td>Page Exists</td>
        <td>10275</td>
        <td>66%</td>
    </tr>
    <tr>
        <td>Common Names</td>
        <td>8509</td>
        <td>55%</td>
    </tr>
    <tr>
        <td>Description</td>
        <td>4771</td>
        <td>31%</td>
    </tr>
</table>

#### **Who has What?**

<table>
    <tr>
        <th>Category</th>
        <th>Number</th>
        <th>% Total</th>
    </tr>
    <tr>
        <td>Both Description and <br>Common Name</td>
        <td>4194</td>
        <td>27%</td>
    </tr>
    <tr>
        <td>Description only</td>
        <td>577</td>
        <td>4%</td>
    </tr>
    <tr>
        <td>Common Name only</td>
        <td>4315</td>
        <td>28%</td>
    </tr>
    <tr>
        <td>Neither</td>
        <td>6415</td>
        <td>41%</td>
    </tr>
</table>