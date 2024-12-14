## 1-Shakespearean Text Generation with GPT-2

  The primary goal of this project was to develop a language model based on the GPT-2 architecture to generate text in the style of **William Shakespeare**.
  The model was trained on a dataset of Shakespearean works to capture the linguistic and stylistic characteristics of the era, such as vocabulary,  
  syntax, and poetic devices like meter and rhyme.

  ![image](https://github.com/user-attachments/assets/1695045a-d311-45cb-948d-2fecb5468871)

**1.1 - Text William Shakespeare** 

   ```
    First Citizen:
    Before we proceed any further, hear me speak.
     
    All:
    Speak, speak.
    
    First Citizen:
    You are all resolved rather to die than to famish?
    
    All:
    Resolved. resolved.
    
    First Citizen:
    First, you know Caius Marcius is chief enemy to the people.
    
    All:
    We know't, we know't.
    
    First Citizen:
    Let us kill him, and we'll have corn at our own price.
    Is't a verdict?
    
    All:
    No more talking on't; let it be done: away, away!
    
    Second Citizen:
    One word, good citizens.
    
    First Citizen:
    We are accounted poor citizens, the patricians good.
    What authority surfeits on would relieve us: if they
    would yield us but the superfluity, while it were
    wholesome, we might guess they relieved us humanely;
    but they think we are too dear: the leanness that
    afflicts us, the object of our misery, is as an
    inventory to particularise their abundance; our
    sufferance is a gain to them Let us revenge this with
    our pikes, ere we become rakes: for the gods know I
    speak this in hunger for bread, not in thirst for revenge.

````
## 2-Data loader: batches of chunks of data

   ```  Python

    x = train_set[:block_size]
    y = train_set[1:block_size+1]
    
    for t in range(block_size):
      context = x[:t+1]
      target  =  y[t]
      print(f"when the input is {context}  - the target: {target}")
```

  ```
    when the input is tensor([18])  - the target: 47
    when the input is tensor([18, 47])  - the target: 56
    when the input is tensor([18, 47, 56])  - the target: 57
    when the input is tensor([18, 47, 56, 57])  - the target: 58
    when the input is tensor([18, 47, 56, 57, 58])  - the target: 1
    when the input is tensor([18, 47, 56, 57, 58,  1])  - the target: 15
    when the input is tensor([18, 47, 56, 57, 58,  1, 15])  - the target: 47
    when the input is tensor([18, 47, 56, 57, 58,  1, 15, 47])  - the target: 58
    
```

## 3 - Hyperparameters Configuration 

  ``` Python
    batch_size = 64 # how many independent sequences will we process in parallel ?
    block_size = 64 # what  is the maximun context length for prediction
    max_iters = 5000
    eval_interval = 300
    learning_rate = 3e-4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_iters = 200
    n_embd = 120
    n_head = 8
    n_layer = 3
    dropout = 0.2
```

**3.1 - Text - Generated without  attention mechanism with after **10.000 iterations****

  ```

   I t that hat pat t hathethat patxbjPiD&Q!athathat t thathat hat t pathathat hat pat t heat's pathat hat's pat's that's t that t t's thathat that's hathat pat het thathathat that 
   hathet pat theat t t t t that's t t theathat tNw$
   I hat thet hathat's t heat hat t t's pat t's pat pat theathethathathat's hethathetheathathat thathathathat's theat's MEO:
   I pat that's thathetheat hat hathat's thathathat's


  ```

  ```

   Loss:0.465354859828949

 ```

**3.2 - Text - Generated with  attention mechanism  after **10.000 iterations****

``` 

    But the suforio daight con for my magy lord;
    How that haven I award names app from bring 
    Thange o' lordsered, that love in Anst to thee mes
    Whost manyal hidy to your Partar! with lears.
    
    CORIOLIUS:
    I dome?
    
    LEONTESS:
    Inde 'First reptured tongence; which and though foll
    You sea, thou haves nood safess implanty trune,
    For delve o'ern have to svance me our of time
    Theis sated corriced funtering in my leasoly.
    Thy comquece with you ceed Hands now,
    Draistirs. Tearrot, that of thee that is nterelt,
    Of mart knost my mad; bustendienter; pont leting up
    my thalks of you thight prelages.
    
    SICHARD:
    I, fear that I see ine on
    And you them penson this of grefecking,
    Brest sperpartime ponys.
    
    VORIOLIO:
    A love tald,
    O, 'Sainst, channot honly charth theree sees.
    
    DUKENCE LIZABET:
    Why boy, Howy no good aiute, saful fa'll Coush hears?
    
    Such:
    Takish'd less, my him to is you, if a what her bast ranks.
    
    MENENIUS:
    No land, nor do: and is brispeak.
    Alo, thed you't nevy thou hish tear the itle.
    I is mele too I proy wirn seed the evisutien earl.
    We with ping all'd nace of law liestrest,
    Thy unford than blost pose kst nobly comoquech!
    Yet rough, bestarniof!
    Arst here me him you are fortue thing dead? one eye
    In think, your hold givedy the re
    Three riaged thy atime bloatings.
    
    WICK:
    Must, broward not the-bost, the do the colice.
    
    LENCESTIO:
    Here thou and earting Shainst Spaint;
    I duke,
    And well but, dided is the to htheeJs ithanges,
    Feargive that you weet of anded as me!
    A fairend, thour doour death! to Remp'd:
    And, see ever then hath.
    Osse, plook usetizer; tiftent foor.
    
    SANLIO:
    No, by my hear I consity, and say slall me,
    And to he and of figred new alliaten not bl I am feed
    
    And.
    Yetak appring matter, not my see rembless.
    
    ADY:
    And had is spy leephat tood your corcirance.
    
    BRINCE:
    Geten chood sarriking a boy, sign the see thours.
    
    GLOUCESTER:
    My wath death are enment, the rinestyfor her
    To boy oun'son equeon that pail:
    Show may liff, is muse with dow.

```

```
 Loss:1.632  
```

## 4 - Results:

  ![image](https://github.com/user-attachments/assets/72c4a661-09aa-4ee8-b350-096cde1c78c0)


## References

- [Andrej Karpathy's gpt-2](https://github.com/karpathy/nanoGPT)
- Deep Learning : Ba

    


     
    
  
