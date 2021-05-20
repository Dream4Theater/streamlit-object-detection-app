## Streamlit Object Detection App

# Installation

```bash
git clone https://github.com/Dream4Theater/streamlit-object-detection-app
cd streamlit-object-detection-app
streamlit run app.py
```

U can run same code above to run on local host.

# Deploy Github

1. U can do this in Github Desktop or u can use bash but first u need to create repository.

![alt text](https://github.com/Dream4Theater/streamlit-object-detection-app/blob/main/images/image1.PNG?raw=true)

2. U need to push ur data to repository.

```bash
cd <your-app-dir>
git init
git add .
git commit -m "first commit"
git branch -M main
git remote add origin git@github.com:<your-app-name>.git
git push -u origin main
```

3. And u are ready to deploy ur app.

# Deploy Streamlit Share

1. So if u wanna deploy your app to streamlit u need to get an invite. Don't worry its not that hard. U can get it from [here.](https://streamlit.io/sharing)

2. And u can easily deploy like this

![alt text](https://github.com/Dream4Theater/streamlit-object-detection-app/blob/main/images/streamlit_sharing_silent.gif?raw=true)

# Deploy Heroku

1. First u need to create an account on heroku and download cli.

2. Then run this code to login heroku

```bash
heroku login
```
3. Then run this code on git bash to create a project and when u create a project u can easily push it from github branch.

```bash
heroku create <your-app-name>
git push heroku main
```

# Conclusion

If everything worked fine sites should be looking like this.
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/dream4theater/streamlit-object-detection-app/main/app.py)
[Heroku App](https://streamlit-object-detection-app.herokuapp.com/)
