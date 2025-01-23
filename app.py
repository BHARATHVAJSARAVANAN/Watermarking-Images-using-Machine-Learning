from flask import Flask, render_template, request, session, redirect, url_for, flash, send_file
import os
import cv2
import numpy as np
from math import log10, sqrt
from skimage.metrics import structural_similarity as ssim
import pywt
from tkinter import filedialog
from tkinter.filedialog import askopenfilename

from datetime import date,time, datetime
import sqlite3 as sql
import os
# Create Database if it doesnt exist
if not os.path.isfile('database.db'):
  conn = sql.connect('database.db')
  conn.execute('CREATE TABLE IF NOT EXISTS Users (Name TEXT NOT NULL, Email TEXT NOT NULL, Password TEXT NOT NULL, Contact INTEGER NOT NULL)')
  conn.close()

app = Flask(__name__)
app.secret_key = "watermarking_secret_key"



UPLOAD_FOLDER = 'static/uploads/'
OUTPUT_FOLDER = 'static/outputs/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

global host_image_path, watermark_image_path, extraction_image_path

# Utility functions for metrics
def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return 100, 0
    max_pixel = 255.0
    psnr = 100 - (20 * log10(max_pixel / sqrt(mse)))
    return psnr, mse

def imageSSIM(normal, embed):
    ssim_value = ssim(normal, embed, data_range=embed.max() - embed.min())
    return ssim_value

# ------------------------------------------------------
# Route for main page


# @app.route('/')
# def home():
#     return render_template('home.html')

@app.route('/')
def home():
    return render_template('home.html', logged_in=session.get('logged_in'))

@app.route('/compatibility.html')
def compatibility():
    return render_template('compatibility.html')

@app.route('/privacy.html')
def privacy():
    return render_template('privacy.html')

@app.route('/terms.html')
def terms():
    return render_template('terms.html')

# @app.route('/terms.html')
# def terms():
#     return render_template('terms.html')

@app.route('/root')
def root():
    session.clear()  # Clear the session to log the user out
    session['logged_out'] = 1
    return redirect(url_for('home.html'))

@app.route('/register', methods=['GET', 'POST'])
def register():
  if request.method == 'POST':
    nm = request.form['nm']
    contact = request.form['contact']
    email = request.form['email']
    password = request.form['password']
         
    with sql.connect("database.db") as con:
      cur = con.cursor()
      #check if User already present
      cur.execute("SELECT Email FROM Users WHERE Email=(?)",[(email)])
      data = cur.fetchall()
      if len(data)>0:
        print('User already exists')
        user_exists=1
      else:
        print("User not found, register new user")
        user_exists=0
        cur.execute("INSERT INTO Users (Name,Email,Password,Contact) VALUES (?,?,?,?)",(nm,email,password,contact) )

        
  return render_template('login.html',user_exists=user_exists, invalid = None, logged_out=None)


@app.route('/login.html',  methods=['GET', 'POST'])
def login():
  invalid = None
  if request.method == 'POST':
    email = request.form['email']
    password = request.form['password']     
    with sql.connect("database.db") as con:
      cur = con.cursor()
      #Validate user credentails from database
      cur.execute("SELECT Email FROM Users WHERE Email=(?) AND Password=(?)",[(email),(password)])
      data = cur.fetchall()
      if len(data)>0:
        print('Login Success')
        # Fetch name of user
        cur.execute("SELECT Name FROM Users WHERE Email=(?) AND Password=(?)",[(email),(password)])
        nm = cur.fetchall()
        nm=nm[0][0]
        # Store User details in Session and log in user
        session['nm'] = nm
        session['email'] = email
        session['logged_out'] = None
        return redirect(url_for('water'))
      else:
        print("Invalid Login")
        invalid=1  
  return render_template('login.html',user_exists=None, invalid = invalid, logged_out=None)

@app.route('/logout')
def logout():
  session.clear()
  session['logged_out']=1
  print('Session Cleared and Logged Out')
  return render_template('home.html')  

#Display Profile
@app.route('/profile')
def profile():
   # If Logged Out, Redirect to Log In page
   if session['logged_out']:
    return render_template('login.html',logged_out=1,user_exists=None, invalid = None)
   nm = session['nm']
   email = session['email']
   with sql.connect("database.db") as con:
    cur = con.cursor()
    # Fetch details of user
    cur.execute("SELECT Contact FROM Users WHERE Email=(?)",[(email)])
    contact = cur.fetchall()
    contact=contact[0][0]

    cur.execute("SELECT Password FROM Users WHERE Email=(?)",[(email)])
    password = cur.fetchall()
    password=password[0][0]
   return render_template("profile.html",nm=nm,email=email,contact=contact,password=password)

@app.route('/edit_profile', methods=['POST'])
def edit_profile():
    if 'logged_out' in session and session['logged_out']:
        return redirect(url_for('login'))  # Redirect to login if user is not logged in
    
    nm = request.form['nm']
    password = request.form['password']
    contact = request.form['contact']
    email = session['email']  # Use email from session as it's the unique identifier

    with sql.connect("database.db") as con:
        cur = con.cursor()
        cur.execute("UPDATE Users SET Name=?, Password=?, Contact=? WHERE Email=?", (nm, password, contact, email))
        con.commit()

    # Update session variables with the new data
    session['nm'] = nm

    flash("Profile updated successfully!")  # Optionally, flash a message confirming update
    return redirect(url_for('profile'))

# -----------------------------------------
@app.route('/water')
def water():
    return render_template('water.html')

# Route to upload images
@app.route('/upload', methods=['POST'])
def upload():
    global host_image_path, watermark_image_path, extraction_image_path
    if 'host_image' in request.files:
        host_image = request.files['host_image']
        host_image_path = os.path.join(UPLOAD_FOLDER, host_image.filename)
        host_image.save(host_image_path)
    
    if 'watermark_image' in request.files:
        watermark_image = request.files['watermark_image']
        watermark_image_path = os.path.join(UPLOAD_FOLDER, watermark_image.filename)
        watermark_image.save(watermark_image_path)

    if 'extraction_image_path' in request.files:
        extraction_image = request.files['extraction_image']
        extraction_image_path = os.path.join(UPLOAD_FOLDER, extraction_image.filename)
        extraction_image.save(extraction_image_path)

    # flash("Images uploaded successfully!")
    # return redirect(url_for('water'))
    # return redirect(url_for('water', message="Images uploaded successfully"))

    flash("Images uploaded successfully!")
    return render_template('water.html', 
                           host_image=host_image_path, 
                           watermark_image=watermark_image_path, 
                           )


# Route for DWT watermarking
@app.route('/run_dwt')
def run_dwt():
    cover_image = cv2.imread(host_image_path, 0)
    watermark_image = cv2.imread(watermark_image_path, 0)

    cover_image = cv2.resize(cover_image, (300, 300))
    watermark_image = cv2.resize(watermark_image, (150, 150))

    cover_image = np.float32(cover_image) / 255
    coeffC = pywt.dwt2(cover_image, 'haar')
    cA, (cH, cV, cD) = coeffC
    watermark_image = np.float32(watermark_image) / 255

    # Embedding
    coeffW = (0.4 * cA + 0.1 * watermark_image, (cH, cV, cD))
    watermarked_image = pywt.idwt2(coeffW, 'haar')
    
    psnr, mse = PSNR(cover_image, watermarked_image)
    ssim_value = imageSSIM(cover_image, watermarked_image)
    watermarked_image_path = os.path.join(OUTPUT_FOLDER, 'dwt_watermarked.png')
    
    
    cv2.imwrite(watermarked_image_path, watermarked_image * 255)

    base_name = os.path.basename(watermarked_image_path)

     # Save the model and coefficients
    np.save(os.path.join('model', base_name), watermarked_image)  # Save the watermarked image model
    np.save(os.path.join('model', f'CA_{base_name}'), cA)  # Save the 'cA' coefficient model

    return render_template('results.html', technique='DWT',
                           psnr=psnr, mse=mse, ssim=ssim_value,
                           watermarked_image=watermarked_image_path)
 

@app.route('/run_svd')
def run_svd():
    cover_image = cv2.imread(host_image_path, 0)
    watermark_image = cv2.imread(watermark_image_path, 0)

    cover_image = cv2.resize(cover_image, (300, 300))
    watermark_image = cv2.resize(watermark_image, (150, 150))

    # Ensure images are in double format for SVD processing
    [m, n] = np.shape(cover_image)
    cover_image = np.double(cover_image)
    watermark_image = np.double(watermark_image)

    # Perform SVD on cover image
    ucvr, wcvr, vtcvr = np.linalg.svd(cover_image, full_matrices=1, compute_uv=1)
    Wcvr = np.zeros((m, n), np.uint8)
    Wcvr[:m, :n] = np.diag(wcvr)
    Wcvr = np.double(Wcvr)

    [x, y] = np.shape(watermark_image)

    # Embed watermark by modifying singular values
    # for i in range(min(x, len(wcvr))):  # Ensure loop does not exceed wcvr's length
    #     wcvr[i] = (wcvr[i] + 0.01 * watermark_image[i % x, i % y]) / 255

    for i in range(x):
        for j in range(y):
            Wcvr[i, j] = (Wcvr[i, j] + 0.01 * watermark_image[i, j]) / 255
    
    # SVD of Wcvr
    u, w, v = np.linalg.svd(Wcvr, full_matrices=1, compute_uv=1)

    # Watermarked image
    S = np.zeros((300, 300), np.uint8)
    S[:m, :n] = np.diag(w)
    wimg = np.matmul(ucvr, np.matmul(S, vtcvr))
    watermarked_image = cv2.normalize(wimg, None, 1.0, 0.0, cv2.NORM_MINMAX)


    # Calculate PSNR, MSE, and SSIM
    psnr, mse = PSNR(cover_image, watermarked_image)
    ssim_value = imageSSIM(cover_image, watermarked_image)

    # Save the watermarked image
    watermarked_image_path = os.path.join(OUTPUT_FOLDER, 'svd_watermarked.png')
    cv2.imwrite(watermarked_image_path, watermarked_image * 255)

    return render_template('results.html', technique='SVD',
                           psnr=psnr, mse=mse, ssim=ssim_value,
                           watermarked_image=watermarked_image_path)



@app.route('/run_extraction', methods=['POST'])
def extraction():
    # Ensure the extraction image is uploaded
    if 'extraction_image' not in request.files:
        flash("No extraction image uploaded!")
        return redirect(url_for('water'))

    extraction_image = request.files['extraction_image']
    extraction_image_path = os.path.join(UPLOAD_FOLDER, extraction_image.filename)
    extraction_image.save(extraction_image_path)

    # Read the watermarked image and the stored cA matrix for extraction
    img = cv2.imread(extraction_image_path, 0)  # Read the uploaded image
    wm_filename = extraction_image.filename
    cA = np.load(f"model/CA_{wm_filename}.npy")  # Load the cA matrix

    # Perform DWT on the watermarked image
    coeffWM = pywt.dwt2(img / 255.0, 'haar')
    hA, (hH, hV, hD) = coeffWM

    # Extract the watermark: reverse the embedding formula used
    extracted = (hA - 0.4 * cA) / 0.1  # Reverse the embedding process
    extracted = np.clip(extracted * 255, 0, 255)  # Scale back to the 0-255 range
    extracted = np.uint8(extracted)

    # Resize extracted watermark to original watermark size
    extracted = cv2.resize(extracted, (200, 200))  # Adjust size to original watermark dimensions

    # Save the extracted watermark
    extracted_path = os.path.join(OUTPUT_FOLDER, 'extracted_watermark.png')
    cv2.imwrite(extracted_path, extracted)

    return render_template('results.html',
                           technique='Extraction (DWT)',
                           extracted_watermark=extracted_path)


if __name__ == "__main__":
    app.run(debug=True)
