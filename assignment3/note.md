## RandomIndexing

Query: Harry Gryffindor chair wand good enter on school

- Metrics
    + Cosine
        ```
        Neighbors for Harry: [('Harry', 3.3306690738754696e-16), ('Hagrid', 0.07176719701776191), ('Snape', 0.07731359532609527), ('Dumbledore', 0.08205713890006394), ('Neville', 0.09163086549184118)]
        Neighbors for Gryffindor: [('Gryffindor', 5.551115123125783e-16), ('Slytherin', 0.11705253885167421), ('school', 0.17978921232135425), ('class', 0.18419073358393123), ('house', 0.18517269616747845)]
        Neighbors for chair: [('chair', 2.220446049250313e-16), ('seat', 0.05255953489712062), ('cauldron', 0.10916551875652558), ('hand', 0.12657402981221966), ('trunk', 0.12998666042217843)]
        Neighbors for wand: [('wand', 0.0), ('head', 0.06323133120539959), ('hand', 0.07187344673162754), ('fingers', 0.09065667314448489), ('leg', 0.09072214969550552)]
        Neighbors for good: [('good', 0.0), ('little', 0.1250934827822905), ('nice', 0.13744030981315425), ('nasty', 0.1383174411817636), ('such', 0.1388672930372863)]
        Neighbors for enter: [('enter', 8.881784197001252e-16), ('leave', 0.10441473127898526), ('retrieve', 0.11271860148775947), ('decor', 0.11890718627031438), ('revert', 0.11890718627031438)]
        Neighbors for on: [('on', 1.1102230246251565e-16), ('into', 0.04383561494428312), ('from', 0.04429414931879805), ('in', 0.04613703880545439), ('through', 0.04639636125393898)]
        Neighbors for school: [('school', 1.1102230246251565e-16), ('castle', 0.06174083370898931), ('house', 0.062493198968932795), ('Burrow', 0.06957213763422687), ('window', 0.07009641238293463)]
        ```

    + L1
        ```
        Neighbors for Harry: [('Harry', 0.0), ('it', 1930862.0), ('he', 2025674.0), ('Ron', 2056814.0), ('Hermione', 2126922.0)]
        Neighbors for Gryffindor: [('Gryffindor', 0.0), ('Slytherin', 99252.0), ('class', 107948.0), ('fire', 109706.0), ('house', 110240.0)]
        Neighbors for chair: [('chair', 0.0), ('seat', 45996.0), ('trunk', 54038.0), ('bag', 57218.0), ('letter', 60414.0)]
        Neighbors for wand: [('wand', 0.0), ('head', 248590.0), ('hand', 264728.0), ('eyes', 280470.0), ('face', 287092.0)]
        Neighbors for good: [('good', 0.0), ('small', 125682.0), ('great', 128738.0), ('such', 141150.0), ('lot', 143194.0)]
        Neighbors for enter: [('enter', 0.0), ('retrieve', 13556.0), ('examine', 13592.0), ('according', 14024.0), ('steal', 14216.0)]
        Neighbors for on: [('on', 0.0), ('into', 881084.0), ('from', 897316.0), ('with', 918256.0), ('for', 1011566.0)]
        Neighbors for school: [('school', 0.0), ('house', 81024.0), ('castle', 86008.0), ('fire', 87256.0), ('place', 90204.0)]
        ```

    + l2
        ```
        Neighbors for Harry: [('Harry', 0.0), ('Ron', 62081.722431002185), ('he', 66406.01441134681), ('Hermione', 66995.99305928676), ('it', 71087.72442552933)]
        Neighbors for Gryffindor: [('Gryffindor', 0.0), ('Slytherin', 3060.9802351534386), ('class', 3228.5160677933754), ('house', 3315.882084755126), ('students', 3335.6861363143867)]
        Neighbors for chair: [('chair', 0.0), ('seat', 1457.3674896881705), ('bag', 1728.8782490389542), ('trunk', 1782.4152153749137), ('body', 1974.371798826148)]
        Neighbors for wand: [('wand', 0.0), ('head', 7661.0462731927155), ('eyes', 8587.939799509542), ('face', 8839.030376687253), ('hand', 8913.891069561036)]
        Neighbors for good: [('good', 0.0), ('small', 3632.279174292637), ('great', 3845.9791991117163), ('gave', 4244.332220738617), ('lot', 4266.997304897203)]
        Neighbors for enter: [('enter', 0.0), ('examine', 404.45518911246523), ('retrieve', 407.47024431239146), ('returning', 411.8154926663153), ('win', 418.7338056570069)]
        Neighbors for on: [('on', 0.0), ('into', 30026.70115080909), ('from', 31548.984516145683), ('at', 32191.873291251628), ('with', 37434.98003205024)]
        Neighbors for school: [('school', 0.0), ('house', 2539.488137400921), ('place', 2661.6690252546427), ('fire', 2715.4108344779065), ('crowd', 2834.1771292563913)]
        ```
    
    I prefer Cosine metric. Cosine metric removes the effect of word frequency on the neighborhood of a word.

- Dimension
    + dim=10, non_zero=8
        ```
        Neighbors for Harry: [('Harry', 0.0), ('Dumbledore', 0.03495456317897028), ('Hagrid', 0.04108410279618202), ('odd', 0.05439005481881043), ('Give', 0.06324471220373584)]
        Neighbors for Gryffindor: [('Gryffindor', 1.1102230246251565e-16), ('excitable', 0.023614676305531712), ('Pointing', 0.06387165569187025), ('civilize', 0.07257136489436045), ('Bones', 0.0834885344361802)]
        Neighbors for chair: [('chair', 0.0), ('detour', 0.020249423802298105), ('teapot', 0.0340932463124064), ('welts', 0.035408084981056076), ('essentials', 0.036441810063119195)]
        Neighbors for wand: [('wand', 0.0), ('recount', 0.038248511904168), ('companions', 0.04349186852849374), ('eyebrows', 0.05181289599025318), ('pestle', 0.0522027938186419)]
        Neighbors for good: [('good', 2.220446049250313e-16), ('providing', 0.025878689500463237), ('small', 0.029277841481604505), ('hearty', 0.030359011407644187), ('diversion', 0.031302599399351094)]
        Neighbors for enter: [('enter', 1.1102230246251565e-16), ('catch', 0.04006173544703262), ('strayed', 0.04070712418428557), ('seize', 0.042032485084754545), ('bury', 0.05751600147984015)]
        Neighbors for on: [('on', 0.0), ('upon', 0.008555738522901146), ('along', 0.013025095448175539), ('through', 0.014311280082255129), ('of', 0.017229892449820206)]
        Neighbors for school: [('school', 2.220446049250313e-16), ('tent', 0.013581611417734063), ('fire', 0.01464237059238549), ('classroom', 0.020523720511445465), ('floor', 0.022269985318934093)]
        ```
        We see that there are unrelated words in the neighborhood of the query word.

    + dim=128, non_zero=8
        ```
        Neighbors for Harry: [('Harry', 0.0), ('Snape', 0.07976315030099901), ('Hagrid', 0.0817404139122645), ('Dumbledore', 0.09066716309101364), ('Sirius', 0.09578839683192109)]
        Neighbors for Gryffindor: [('Gryffindor', 0.0), ('Slytherin', 0.13147040765150808), ('House', 0.19128929404305273), ('house', 0.19493170743284383), ('school', 0.1985968859097258)]
        Neighbors for chair: [('chair', 0.0), ('seat', 0.05897038016244416), ('cauldron', 0.10306773750731013), ('cupboard', 0.11444948521275178), ('trunk', 0.13594584908408847)]
        Neighbors for wand: [('wand', 0.0), ('head', 0.07105365345711179), ('fingers', 0.0851073967475191), ('nose', 0.08760566573227391), ('hand', 0.08787846707274238)]
        Neighbors for good: [('good', 0.0), ('such', 0.11095791376784392), ('nice', 0.13281952187292556), ('nasty', 0.1457661699750712), ('big', 0.14820307296624224)]
        Neighbors for enter: [('enter', 1.1102230246251565e-16), ('owing', 0.08473167588338903), ('return', 0.10339356727946891), ('break', 0.10392380619513264), ('pass', 0.1047409312755061)]
        Neighbors for on: [('on', 0.0), ('into', 0.030306574369180272), ('from', 0.03573787831730335), ('in', 0.03898871727863784), ('through', 0.03940672616728935)]
        Neighbors for school: [('school', 0.0), ('castle', 0.05990508239404302), ('house', 0.0677784029302495), ('boys', 0.0704229157443843), ('fire', 0.07160560651676351)]
        ```
        The performance is much better and is about the same as when dim=1000.

- Left window size
    + ls=0
        ```
        Neighbors for Harry: [('Harry', 0.0), ('Hermione', 0.12444179353522866), ('Weasley', 0.16583662910945596), ('Dumbledore', 0.16740723588470718), ('Hagrid', 0.1856183058785823)]
        Neighbors for Gryffindor: [('Gryffindor', 0.0), ('Slytherin', 0.26605916090066295), ('Ravenclaw', 0.3304910321156771), ('North', 0.4131012658721299), ('Astronomy', 0.4424857117676364)]
        Neighbors for chair: [('chair', 0.0), ('table', 0.11322382197076197), ('floor', 0.1367252269320074), ('fire', 0.1407939607607639), ('window', 0.14569219111959575)]
        Neighbors for wand: [('wand', 2.220446049250313e-16), ('laughter', 0.09112753532112428), ('fire', 0.11830899874230583), ('window', 0.12272970850150322), ('table', 0.12489745781637185)]
        Neighbors for good: [('good', 0.0), ('bad', 0.2651717202860481), ('stupid', 0.2757481131321988), ('here', 0.2846520174633904), ('me', 0.28775571408196665)]
        Neighbors for enter: [('enter', 3.3306690738754696e-16), ('through', 0.13907768600956405), ('from', 0.1417703618077989), ('misreading', 0.15225812468151778), ('etch', 0.15225812468151778)]
        Neighbors for on: [('on', 0.0), ('from', 0.02038105274763491), ('onto', 0.030992956482835066), ('over', 0.032813640509986586), ('of', 0.03898592445163296)]
        Neighbors for school: [('school', 0.0), ('house', 0.13442186081185215), ('again', 0.13766490718005342), ('head', 0.15368903586993998), ('castle', 0.15804987727079423)]
        ```

    + ls=10
        ```
        Neighbors for Harry: [('Harry', 0.0), ('Snape', 0.045727573856170545), ('Neville', 0.050899256605097865), ('Hagrid', 0.053870336369435945), ('Malfoy', 0.055387126396371555)]
        Neighbors for Gryffindor: [('Gryffindor', 0.0), ('Slytherin', 0.04523999122632749), ('house', 0.05962334144043813), ('way', 0.06630704206755145), ('most', 0.07079434709315902)]
        Neighbors for chair: [('chair', 0.0), ('seat', 0.0423873924182574), ('bag', 0.07546070933069504), ('and', 0.07662462713159413), ('cauldron', 0.07736303492812091)]
        Neighbors for wand: [('wand', 0.0), ('hand', 0.03744897299195693), ('head', 0.04014600396290746), ('fingers', 0.05682796774923471), ('arm', 0.06103912989194593)]
        Neighbors for good: [('good', 2.220446049250313e-16), ('little', 0.06615282075727191), ('very', 0.0752173096737202), ('such', 0.07688964045379798), ('nice', 0.08007713232167457)]
        Neighbors for enter: [('enter', 1.1102230246251565e-16), ('take', 0.10918921980544982), ('break', 0.12146421426992626), ('leave', 0.12705413248907726), ('read', 0.13313622520491708)]
        Neighbors for on: [('on', 2.220446049250313e-16), ('from', 0.015472152233157788), ('in', 0.016541633858245253), ('over', 0.020110806487470656), ('of', 0.028211990032141476)]
        Neighbors for school: [('school', 0.0), ('house', 0.04213343845192086), ('place', 0.05576802495942279), ('next', 0.0602736651914666), ('family', 0.06094554779947747)]
        ```
    When the left window size is too small or too large, the performance becomes worse.

## Word2vec

The Word2vec embeddings give less intuitive results than random indexing.