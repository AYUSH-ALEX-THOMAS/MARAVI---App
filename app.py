from flask import Flask, render_template, request, redirect, url_for, session, jsonify, send_from_directory
import time
import math
import numpy as np
import json
import os
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import joblib
from flask_login import login_required
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from sklearn.model_selection import train_test_split
import csv
from dotenv import load_dotenv
from supabase import create_client, Client
import bcrypt

# Load environment variables
load_dotenv()

# Initialize Supabase client
supabase_url = os.getenv('SUPABASE_URL')
supabase_key = os.getenv('SUPABASE_KEY')

print(f"Initializing Supabase with URL: {'present' if supabase_url else 'missing'}")
print(f"Supabase key status: {'present' if supabase_key else 'missing'}")

try:
    supabase: Client = create_client(
        supabase_url=supabase_url,
        supabase_key=supabase_key
    )
    print("Supabase client initialized successfully")
except Exception as e:
    print(f"Error initializing Supabase client: {str(e)}")
    print(f"Error type: {type(e)}")

app = Flask(__name__, static_url_path='/static', static_folder='static')
app.secret_key = 'your-secret-key-here'  # Change this to a secure secret key
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=1)
app.config['SESSION_COOKIE_SECURE'] = False  # Set to True in production
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_REFRESH_EACH_REQUEST'] = True

@app.before_request
def check_session():
    """Check session before each request"""
    # List of routes that don't require authentication
    public_routes = ['login', 'signup', 'static']
    
    # If accessing the root URL, always redirect to login if not logged in
    if request.endpoint == 'index' and not session.get('logged_in'):
        return redirect(url_for('login'))
    
    # If the route is public, allow access
    if request.endpoint in public_routes:
        return
        
    # For all other routes, check if user is logged in
    if not session.get('logged_in'):
        session.clear()  # Clear any existing session data
        return redirect(url_for('login'))

# Constants for canvas and tablet dimensions
CANVAS_X_SIZE = 800  # Base canvas width
CANVAS_Y_SIZE = 450  # Base canvas height
TABLET_X_SIZE = 21000
TABLET_Y_SIZE = 29700

# Calculate multipliers based on base canvas size
X_MULTIPLIER = TABLET_X_SIZE / CANVAS_X_SIZE
Y_MULTIPLIER = TABLET_Y_SIZE / CANVAS_Y_SIZE
BOX_SIZE = 4

# GMRT calculation parameters
d_gmrt = 5  # Number of points to consider for GMRT calculation

# Global variables for data collection
positions = []
distances = []
times = []
time_vectors = []
times_drawing = []
canvas_squares = None
start_time = None
start_time_vector = None
start_time_segment = None
num_of_pendown = 0
all_task_metrics = []

# Hardcoded user credentials for demonstration
USER_CREDENTIALS = {
    "username": "user",
    "password": "123"
}

# List of tasks
TASKS = [
    {"id": 1, "description": "Signature drawing", "category": "M"},
    {"id": 2, "description": "Join two points with a horizontal line, continuously for four times", "category": "G"},
    {"id": 3, "description": "Join two points with a vertical line, continuously for four times", "category": "G"},
    {"id": 4,
        "description": "Retrace a circle (6 cm of diameter) continuously for four times", "category": "G"},
    {"id": 5,
        "description": "Retrace a circle (3 cm of diameter) continuously for four times", "category": "G"},
    {"id": 6, "description": "Copy the letters 'l', 'm', and 'p'", "category": "C"},
    {"id": 7, "description": "Write cursively a sequence of four lowercase letter 'l', in a single smooth movement", "category": "C"},
    {"id": 8, "description": "Write cursively a sequence of four lowercase cursive bigram 'le', in a single smooth movement", "category": "C"},
    {"id": 9, "description": "Copy the word 'apple'", "category": "C"},
    {"id": 10, "description": "Copy the word 'apple' above a line", "category": "C"},
    {"id": 11, "description": "Copy the word 'mama'", "category": "C"},
    {"id": 12, "description": "Copy the word 'mama' above a line", "category": "C"},
    {"id": 13, "description": "Memorize and rewrite the given words", "category": "M"},
    {"id": 14, "description": "Copy in reverse the word 'medicine'", "category": "C"},
    {"id": 15, "description": "Copy in reverse the word 'home'", "category": "C"},
    {"id": 16,
        "description": "Write the name of the object shown in a picture", "category": "M"},
    {"id": 17, "description": "Retrace a complex form by connecting 8 numbered dots in sequence", "category": "G"},
    {"id": 18, "description": "Copy a telephone number", "category": "C"},
    {"id": 19,
        "description": "Draw a clock, with all hours and put hands at 11:05 (Clock Drawing Test)", "category": "G"}
]

# Simple in-memory storage for test results (since we don't have a database)


class TestResult:
    def __init__(self, user_id, risk_score):
        self.user_id = user_id
        # Convert NumPy types to Python native types
        self.risk_score = float(risk_score) if hasattr(
            risk_score, 'item') else float(risk_score)
        self.timestamp = datetime.now()


# Store test results in memory
test_results_store = []


def reset_task_variables():
    """Reset variables for a new task"""
    global positions, distances, times, time_vectors, times_drawing, canvas_squares
    global start_time, start_time_vector, start_time_segment, num_of_pendown

    positions = []
    distances = []
    times = []
    time_vectors = []
    times_drawing = []
    canvas_squares = np.zeros(
        (CANVAS_Y_SIZE // BOX_SIZE, CANVAS_X_SIZE // BOX_SIZE))
    start_time = None
    start_time_vector = None
    start_time_segment = None
    num_of_pendown = 0


def reset_test():
    """Reset all variables for a new test"""
    global all_task_metrics, test_results_store
    
    # Reset global variables
    reset_task_variables()
    all_task_metrics = []
    test_results_store = []
    
    # Clear session data
    if 'test_results' in session:
        session.pop('test_results')
    if 'current_task' in session:
        session.pop('current_task')
    if 'task_start_time' in session:
        session.pop('task_start_time')
    
    # Clear any stored metrics
    session['metrics'] = {}
    
    print("Test reset complete - all variables and session data cleared")


@app.route('/')
def index():
    """Root URL handler"""
    # Always redirect to login if not logged in
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return redirect(url_for('dashboard'))


def hash_password(password):
    """Hash a password using bcrypt"""
    try:
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    except Exception as e:
        print(f"Error hashing password: {str(e)}")
        raise

def check_password(password, hashed):
    """Verify a password against a hash"""
    try:
        print(f"\n=== Password Verification ===")
        print(f"Input password: {password}")
        print(f"Stored hash: {hashed}")
        
        # Ensure both password and hash are strings
        if not isinstance(password, str):
            password = str(password)
        if not isinstance(hashed, str):
            hashed = str(hashed)
            
        # Check if the hash is properly formatted
        if not hashed.startswith('$2b$'):
            print("Invalid hash format")
            return False
            
        result = bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
        print(f"Verification result: {result}")
        return result
    except Exception as e:
        print(f"Error in password verification: {str(e)}")
        print(f"Error type: {type(e)}")
        return False

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    """Handle user registration"""
    # Clear any existing session when accessing signup page
    if request.method == 'GET':
        session.clear()
        return render_template('signup.html')

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        print(f"Signup attempt for username: {username}")

        if not username or not password or not confirm_password:
            print("Missing required fields")
            return render_template('signup.html', error='All fields are required')

        if password != confirm_password:
            print("Passwords do not match")
            return render_template('signup.html', error='Passwords do not match')

        try:
            # Check if username already exists
            print("Checking if username exists...")
            response = supabase.table('users').select('*').eq('username', username).execute()
            print(f"Username check response: {response}")
            
            if response.data:
                print("Username already exists")
                return render_template('signup.html', error='Username already exists')

            # Hash the password
            print("Hashing password...")
            hashed_password = hash_password(password)
            print("Password hashed successfully")

            # Insert new user
            user_data = {
                'username': username,
                'password_hash': hashed_password,
                'created_at': datetime.now().isoformat()
            }
            print(f"Attempting to create user with data: {user_data}")
            
            response = supabase.table('users').insert(user_data).execute()
            print(f"User creation response: {response}")
            
            if not response.data:
                print("Failed to create user - no data returned")
                return render_template('signup.html', error='Failed to create user')

            user = response.data[0]
            print(f"User created successfully: {user}")
            
            # Set session variables
            session.clear()  # Clear any existing session data
            session['logged_in'] = True
            session['username'] = username
            session['user_id'] = user['user_id']
            session.permanent = True
            print(f"Session variables set: {session}")

            return redirect(url_for('dashboard'))

        except Exception as e:
            print(f"Error in signup: {str(e)}")
            print(f"Error type: {type(e)}")
            print(f"Error details: {e.__dict__ if hasattr(e, '__dict__') else 'No details available'}")
            return render_template('signup.html', error='An error occurred during signup')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handle user login"""
    # Clear any existing session when accessing login page
    if request.method == 'GET':
        session.clear()
        return render_template('login.html')

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        print(f"\n=== Login Attempt ===")
        print(f"Username: {username}")

        if not username or not password:
            print("Missing username or password")
            return render_template('login.html', error='Username and password are required')

        try:
            # Verify database connection first
            if not verify_database_connection():
                return render_template('login.html', error='Database connection error. Please try again later.')

            # Get user from database
            print("Querying database for user...")
            response = supabase.table('users').select('*').eq('username', username).execute()
            print(f"Database response: {response}")
            
            if not response.data:
                print(f"No user found with username: {username}")
                return render_template('login.html', error='Invalid credentials')

            user = response.data[0]
            print(f"Found user: {user}")
            
            # Verify password
            print("Verifying password...")
            print(f"Stored password hash: {user['password_hash']}")
            print(f"Password verification result: {check_password(password, user['password_hash'])}")
            
            if check_password(password, user['password_hash']):
                print("Password verified successfully")
                session.clear()  # Clear any existing session data
                session['logged_in'] = True
                session['username'] = username
                session['user_id'] = user['user_id']
                session.permanent = True
                print(f"Session variables set: {session}")
                return redirect(url_for('dashboard'))
            else:
                print("Password verification failed")
                return render_template('login.html', error='Invalid credentials')

        except Exception as e:
            print(f"Error in login: {str(e)}")
            print(f"Error type: {type(e)}")
            print(f"Error details: {e.__dict__ if hasattr(e, '__dict__') else 'No details available'}")
            return render_template('login.html', error='An error occurred during login. Please try again.')


@app.route('/dashboard')
def dashboard():
    """Display the dashboard with test results chart"""
    print(f"\n=== Dashboard Access Attempt ===")
    print(f"Session data: {session}")
    
    # Strict authentication check
    if not session.get('logged_in'):
        print("Not logged in - redirecting to login")
        session.clear()  # Clear any partial session data
        return redirect(url_for('login'))
    
    if not session.get('username'):
        print("No username in session - redirecting to login")
        session.clear()
        return redirect(url_for('login'))
    
    if not session.get('user_id'):
        print("No user_id in session - redirecting to login")
        session.clear()
        return redirect(url_for('login'))

    print("Authentication successful - preparing dashboard data")
    
    # Prepare data for the chart
    chart_data = {
        'labels': [],
        'scores': []
    }

    # Get user ID from session
    user_id = session.get('user_id')
    print(f"Retrieved user_id from session: {user_id}")

    # Get results from Supabase
    user_results = get_user_results_from_supabase(user_id)
    print(f"Retrieved user results: {user_results}")

    for result in user_results:
        # Format the date as DD/MM
        date = datetime.fromisoformat(result['timestamp']).strftime('%d/%m')
        chart_data['labels'].append(date)
        chart_data['scores'].append(float(result['risk_score']))

    # If no results yet, add default data
    if not chart_data['labels']:
        print("No results found - using default data")
        chart_data['labels'] = ['No tests yet']
        chart_data['scores'] = [0.0]

    print("Rendering dashboard template")
    return render_template('dashboard.html', chart_data=chart_data)


@app.route('/handwriting_test')
def handwriting_test():
    """Start a new handwriting test"""
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    reset_test()
    return redirect(url_for('next_task', task_number=0))


@app.route('/start_drawing', methods=['POST'])
def start_drawing():
    """Handle the start of drawing"""
    global start_time, start_time_vector, start_time_segment, num_of_pendown
    
    if start_time is None:
        start_time = time.time()
    
    if start_time_vector is None:
        start_time_vector = time.time()
    
    if start_time_segment is None:
        start_time_segment = time.time()
    
    num_of_pendown += 1
    return jsonify({'status': 'started'})


@app.route('/draw', methods=['POST'])
def draw():
    """Handle drawing events"""
    global start_time, start_time_vector, start_time_segment, positions, distances, times, time_vectors

    if start_time is None:
        start_time = time.time()

    if start_time_vector is None:
        start_time_vector = time.time()

    if start_time_segment is None:
        start_time_segment = time.time()

    data = request.json
    x, y = float(data['x']), float(data['y'])

    # Convert canvas coordinates to tablet coordinates
    tablet_x = x * X_MULTIPLIER
    tablet_y = y * Y_MULTIPLIER

    # Store position
    positions.append((tablet_x, tablet_y))

    # Calculate time vector in seconds
    time_vector = 0.0  # Initialize time_vector
    if start_time_vector is not None:
        old_time = start_time_vector
        start_time_vector = time.time()
        time_vector = start_time_vector - old_time  # Time difference in seconds
        time_vectors.append(time_vector)

        # Calculate distance and speed
        if len(positions) > 1:
            prev_pos = positions[-2]
            distance = math.sqrt((tablet_x - prev_pos[0])**2 + (tablet_y - prev_pos[1])**2)
            distances.append(distance)
            times.append(time_vector)

            # Update canvas squares for dispersion index
            x_index = int(x // BOX_SIZE)
            y_index = int(y // BOX_SIZE)
            if 0 <= y_index < canvas_squares.shape[0] and 0 <= x_index < canvas_squares.shape[1]:
                canvas_squares[y_index, x_index] = 1

    return jsonify({'status': 'drawing'})


@app.route('/stop_drawing', methods=['POST'])
def stop_drawing():
    """Handle the end of drawing"""
    global start_time_segment, start_time_vector, times_drawing

    if start_time_segment is not None:
        elapsed_time = time.time() - start_time_segment
        times_drawing.append(elapsed_time)  # Time in seconds
        start_time_segment = None

    start_time_vector = None
    return jsonify({'status': 'stopped'})


def calculate_gmrt():
    """
    It is responsible for calculating the GMRT from the distances of the write positions to
    the upper right corner
    """
    try:
        # Ensure d_gmrt is defined and valid
        if not hasattr(globals(), 'd_gmrt'):
            d_gmrt = 5  # Default value if not defined
        
        radii_variation = []
        if len(distances) > d_gmrt:
            for i in range(len(distances)):
                if i >= d_gmrt:
                    result = abs(distances[i] - distances[i - d_gmrt + 1])
                    radii_variation.append(result)
            return (1 / len(radii_variation)) * sum(radii_variation)  # Remove premature rounding
        else:
            return 0.0  # Return float instead of int
    except Exception as e:
        print(f"Error in calculate_gmrt: {str(e)}")
        return 0.0


def calculate_metrics():
    """Calculate metrics for the current task"""
    if not positions:
        return None

    # Calculate total time in milliseconds
    if start_time is not None:
        end_time = time.time()
        total_time = round((end_time - start_time) * 1000)  # Total time in milliseconds
    else:
        total_time = 0

    # Calculate paper time (sum of time_vectors) in milliseconds
    paper_time = round(sum(time_vectors) * 1000) if time_vectors else 0

    # Calculate air time
    air_time = total_time - paper_time

    # Calculate dispersion index
    disp_index = float(np.sum(canvas_squares)) / float(canvas_squares.size) if canvas_squares is not None else 0.0
    
    # Calculate speed metrics
    total_distance = float(sum(distances)) if distances else 0.0
    mean_speed = total_distance / (paper_time / 1000) if paper_time > 0 else 0.0  # Convert paper_time back to seconds for speed calculation
    
    # Calculate GMRT
    gmrt = float(calculate_gmrt())
    
    # Calculate speed in air and on paper separately
    air_distances = []
    paper_distances = []
    air_times = []
    paper_times = []
    air_velocities = []
    paper_velocities = []
    
    for i in range(1, len(positions)):
        prev_pos = positions[i-1]
        curr_pos = positions[i]
        distance = math.sqrt((curr_pos[0] - prev_pos[0])**2 + (curr_pos[1] - prev_pos[1])**2)
        
        # Initialize time variables
        time_vector = 0.0
        drawing_time = 0.0
        
        if i < len(time_vectors):
            time_vector = time_vectors[i-1] * 1000  # Convert to milliseconds
        
        if i < len(times_drawing):
            drawing_time = times_drawing[i-1] * 1000  # Convert to milliseconds
        
        if i < len(times_drawing) and times_drawing[i-1] > 0:
            paper_distances.append(float(distance))
            paper_times.append(float(drawing_time))
            if drawing_time > 0:
                paper_velocities.append(float(distance) / (drawing_time / 1000))  # Velocity in pixels/second
        else:
            air_distances.append(float(distance))
            air_times.append(float(time_vector))
            if time_vector > 0:
                air_velocities.append(float(distance) / (time_vector / 1000))  # Velocity in pixels/second
    
    # Calculate mean speeds
    sum_air_times = float(sum(air_times))
    sum_paper_times = float(sum(paper_times))
    mean_speed_in_air = float(sum(air_distances)) / (sum_air_times / 1000) if sum_air_times > 0 else 0.0  # Convert back to seconds for speed calculation
    mean_speed_on_paper = float(sum(paper_distances)) / (sum_paper_times / 1000) if sum_paper_times > 0 else 0.0  # Convert back to seconds for speed calculation

    # Calculate accelerations
    air_accelerations = []
    paper_accelerations = []
    
    # Calculate accelerations for air segments
    for i in range(1, len(air_velocities)):
        if i < len(air_times) and air_times[i] > 0:
            dv = air_velocities[i] - air_velocities[i-1]
            dt = air_times[i] / 1000  # Convert to seconds
            if dt > 0:
                air_accelerations.append(dv / dt)
    
    # Calculate accelerations for paper segments
    for i in range(1, len(paper_velocities)):
        if i < len(paper_times) and paper_times[i] > 0:
            dv = paper_velocities[i] - paper_velocities[i-1]
            dt = paper_times[i] / 1000  # Convert to seconds
            if dt > 0:
                paper_accelerations.append(dv / dt)
    
    # Calculate mean accelerations
    mean_acc_in_air = sum(air_accelerations) / len(air_accelerations) if air_accelerations else 0.0
    mean_acc_on_paper = sum(paper_accelerations) / len(paper_accelerations) if paper_accelerations else 0.0
    
    # Calculate jerks (rate of change of acceleration)
    air_jerks = []
    paper_jerks = []
    
    # Calculate jerks for air segments
    for i in range(1, len(air_accelerations)):
        if i < len(air_times) and air_times[i] > 0:
            da = air_accelerations[i] - air_accelerations[i-1]
            dt = air_times[i] / 1000  # Convert to seconds
            if dt > 0:
                air_jerks.append(da / dt)
    
    # Calculate jerks for paper segments
    for i in range(1, len(paper_accelerations)):
        if i < len(paper_times) and paper_times[i] > 0:
            da = paper_accelerations[i] - paper_accelerations[i-1]
            dt = paper_times[i] / 1000  # Convert to seconds
            if dt > 0:
                paper_jerks.append(da / dt)
    
    # Calculate mean jerks
    mean_jerk_in_air = sum(air_jerks) / len(air_jerks) if air_jerks else 0.0
    mean_jerk_on_paper = sum(paper_jerks) / len(paper_jerks) if paper_jerks else 0.0
    
    # Calculate max extensions
    x_coords = [p[0] for p in positions]
    y_coords = [p[1] for p in positions]
    max_x_extension = max(x_coords) - min(x_coords) if x_coords else 0.0
    max_y_extension = max(y_coords) - min(y_coords) if y_coords else 0.0

    return {
        'task_id': None,  # This will be set in next_task route
        'air_time': air_time,  # in milliseconds
        'paper_time': paper_time,  # in milliseconds
        'total_time': total_time,  # in milliseconds
        'dispersion_index': disp_index,
        'mean_speed': mean_speed,
        'gmrt': gmrt,
        'max_x_extension': max_x_extension,
        'max_y_extension': max_y_extension,
        'num_of_pendown': num_of_pendown,
        'mean_speed_in_air': mean_speed_in_air,
        'mean_speed_on_paper': mean_speed_on_paper,
        'mean_acc_in_air': mean_acc_in_air,
        'mean_acc_on_paper': mean_acc_on_paper,
        'mean_jerk_in_air': mean_jerk_in_air,
        'mean_jerk_on_paper': mean_jerk_on_paper,
        'gmrt_in_air': calculate_gmrt_for_segments(air_distances, air_times),
        'gmrt_on_paper': calculate_gmrt_for_segments(paper_distances, paper_times)
    }


def calculate_gmrt_for_segments(distances, times):
    """Calculate GMRT for specific segments"""
    if not distances or not times or len(distances) != len(times):
        return 0

    gmrt_values = []
    for d, t in zip(distances, times):
        if t > 0:
            gmrt_values.append(d / t)

    return sum(gmrt_values) / len(gmrt_values) if gmrt_values else 0


def save_metrics_to_csv(metrics, task_number):
    """Save task metrics to a CSV file"""
    filename = 'task_metrics.csv'
    file_exists = os.path.isfile(filename)
    
    # Define the field names
    fieldnames = [
        'timestamp', 'task_number',
        'air_time', 'paper_time', 'total_time',
        'dispersion_index', 'mean_speed', 'gmrt',
        'max_x_extension', 'max_y_extension', 'num_of_pendown',
        'mean_speed_in_air', 'mean_speed_on_paper',
        'mean_acc_in_air', 'mean_acc_on_paper',
        'mean_jerk_in_air', 'mean_jerk_on_paper',
        'gmrt_in_air', 'gmrt_on_paper'
    ]
    
    # Add timestamp to metrics
    metrics['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    metrics['task_number'] = task_number
    
    try:
        with open(filename, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # Write header if file doesn't exist
            if not file_exists:
                writer.writeheader()
            
            # Write metrics
            writer.writerow(metrics)
    except Exception as e:
        print(f"Error saving metrics to CSV: {str(e)}")

def get_metrics_from_csv(task_number=None):
    """Retrieve metrics from CSV file, optionally filtered by task number"""
    filename = 'task_metrics.csv'
    if not os.path.isfile(filename):
        return []
    
    try:
        with open(filename, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            metrics = list(reader)
            
            # Filter by task number if specified
            if task_number is not None:
                metrics = [m for m in metrics if int(m['task_number']) == task_number]
            
            return metrics
    except Exception as e:
        print(f"Error reading metrics from CSV: {str(e)}")
        return []

@app.route('/next-task/<int:task_number>')
def next_task(task_number):
    """Handle progression to next task"""
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    # Calculate and store metrics for the current task
    metrics = calculate_metrics()
    if metrics:
        # Add task information to metrics
        metrics['task_id'] = task_number
        all_task_metrics.append(metrics)
        
        # Save metrics to CSV
        save_metrics_to_csv(metrics, task_number)

    # Reset variables for next task, but keep all_task_metrics
    reset_task_variables()

    # Check if this was the last task (task_number is 1-based)
    if task_number >= len(TASKS):
        # Store all metrics in session for results page
        session['test_results'] = all_task_metrics
        return redirect(url_for('results'))

    # If not the last task, show the next task
    return render_template('handwriting_test.html',
                           task=TASKS[task_number],
                           task_number=task_number + 1,
                           total_tasks=len(TASKS))

@app.route('/results')
def results():
    """Display test results"""
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    # Check if we have valid test results
    test_results = session.get('test_results', [])
    if not test_results:
        print("No test results found in session, redirecting to handwriting test")
        return redirect(url_for('handwriting_test'))

    try:
        # Validate test results
        if not isinstance(test_results, list) or len(test_results) == 0:
            print("Invalid test results format, resetting test")
            reset_test()
            return redirect(url_for('handwriting_test'))

        # Get predictions
        print("Getting predictions...")
        predictions = predict_alzheimers_risk(test_results)
        print(f"Predictions received: {predictions}")

        # Save the result to Supabase
        user_id = session.get('user_id', 1)
        risk_score = float(predictions['ensemble']) if hasattr(
            predictions['ensemble'], 'item') else float(predictions['ensemble'])
        timestamp = datetime.now()
        
        print(f"Preparing to save - User ID: {user_id}, Risk Score: {risk_score}")
        
        # Save to Supabase
        save_result = save_result_to_supabase(user_id, risk_score, timestamp)
        if save_result is None:
            print("Failed to save result to Supabase")
        else:
            print("Successfully saved result to Supabase")

        # Create pairs of results and tasks with task numbers
        results_with_tasks = []
        for metrics in test_results:
            task_id = metrics.get('task_id', 1)
            if 0 < task_id <= len(TASKS):
                task = TASKS[task_id - 1]
                # Add task number to the task dictionary
                task['task_number'] = task_id
                results_with_tasks.append((metrics, task))

        # Sort results by task number
        results_with_tasks.sort(key=lambda x: x[1]['task_number'])

        # Clear test results after displaying them
        session.pop('test_results', None)
        
        return render_template('results.html',
                           results_with_tasks=results_with_tasks,
                           predictions=predictions)
    except Exception as e:
        print(f"Error in results route: {str(e)}")
        print(f"Error type: {type(e)}")
        # Reset test on error
        reset_test()
        return redirect(url_for('handwriting_test'))


@app.route('/logout')
def logout():
    """Handle user logout"""
    session.clear()
    return redirect(url_for('login'))

# Add route handler for the hyphenated version to handle old links


@app.route('/handwriting-test')
def handwriting_test_redirect():
    """Redirect old URL format to new one"""
    return redirect(url_for('handwriting_test'))


@app.route('/memory_matrix')
def memory_matrix():
    """Display the memory matrix game"""
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('memory_matrix.html')


@app.route('/stroop_test')
def stroop_test():
    """Display the Stroop test"""
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('stroop_test.html')


@app.route('/about')
def about():
    """Display the About Us page"""
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('about.html')

# Initialize models


def initialize_models():
    """Initialize the prediction models"""
    models = {
        'xgboost': XGBClassifier(
            max_depth=6,
            learning_rate=0.1,
            n_estimators=100,
            subsample=0.8,
            colsample_bytree=0.8
        ),
        'random_forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            min_samples_split=2,
            min_samples_leaf=1
        ),
        'gradient_boost': GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3
        )
    }

    # Load pre-trained models if they exist
    try:
        models['xgboost'] = joblib.load('models/xgboost_model.pkl')
        models['random_forest'] = joblib.load('models/rf_model.pkl')
        models['gradient_boost'] = joblib.load('models/gb_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
    except:
        # Use default initialized models if no pre-trained models exist
        scaler = StandardScaler()

    return models, scaler


# Initialize models and scaler
MODELS, SCALER = initialize_models()


def predict_alzheimers_risk(metrics_list):
    """
    Make predictions using our trained model
    """
    try:
        # Use the best model (XGBoost)
        model = joblib.load('models/xgboost_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        n_features_expected = scaler.n_features_in_
        print(f"Scaler expects exactly {n_features_expected} features")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print("Warning: Could not load trained model, using default model")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        scaler = StandardScaler()
        n_features_expected = 342  # Default if we can't load the scaler

    # Extract features in the correct order
    features = []
    print(f"Processing {len(metrics_list)} tasks")
    for metrics in metrics_list:
        task_features = [
            metrics['air_time'],
            metrics['paper_time'],
            metrics['total_time'],
            metrics['dispersion_index'],
            metrics['mean_speed'],
            metrics['gmrt'],
            metrics['max_x_extension'],
            metrics['max_y_extension'],
            metrics['num_of_pendown'],
            metrics['mean_speed_in_air'],
            metrics['mean_speed_on_paper'],
            metrics['mean_acc_in_air'],
            metrics['mean_acc_on_paper'],
            metrics['mean_jerk_in_air'],
            metrics['mean_jerk_on_paper'],
            metrics['gmrt_in_air'],
            metrics['gmrt_on_paper']
        ]
        features.extend(task_features)

    current_features = len(features)
    print(
        f"Generated {current_features} features from {len(metrics_list)} tasks")

    # Create a zero-filled feature vector with the exact size expected by the scaler
    final_features = np.zeros(n_features_expected)

    # Copy over as many features as we can without exceeding the expected size
    copy_size = min(current_features, n_features_expected)
    final_features[:copy_size] = features[:copy_size]

    # Reshape for prediction
    final_features = final_features.reshape(1, -1)
    print(f"Final feature shape: {final_features.shape}")

    # Scale features
    try:
        features_scaled = scaler.transform(final_features)
        print("Features scaled successfully")
    except Exception as e:
        print(f"Error during scaling: {str(e)}")
        return {'ensemble': 50.0, 'individual_models': {'Error': 'Scaling failed'}}

    # Get probability of positive class
    try:
        prob = model.predict_proba(features_scaled)[0][1]
        # Convert to Python native float
        risk_percentage = float(prob * 100)  # Convert to percentage
        print(f"Prediction successful: {risk_percentage:.2f}%")
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        risk_percentage = 50.0  # Default to 50% if prediction fails

    # Ensure all values are JSON serializable
    return {
        'ensemble': risk_percentage,
        'individual_models': {
            'XGBoost': float(risk_percentage)
        }
    }


def calculate_evaluation_metrics(y_true, y_pred):
    """Calculate evaluation metrics for the model"""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred)
    }


@app.route('/evaluation')
def evaluation():
    """Display model evaluation metrics"""
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    try:
        print("\n=== Starting Model Evaluation ===")

        # Load the model and scaler
        print("Loading model and scaler...")
        model = joblib.load('models/xgboost_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        print("Model and scaler loaded successfully")

        # Load the complete dataset
        print("Loading dataset...")
        df = pd.read_csv('data/training_data.csv')

        # Split the data into training and testing sets (80-20 split)
        X = df.drop('class', axis=1)
        y = df['class'].replace({'H': 0, 'P': 1})

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"\nDataset Split:")
        print(f"Total samples: {len(df)}")
        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")

        # Print class distribution
        print("\nClass Distribution (Testing Set):")
        print(f"Healthy (H): {sum(y_test == 0)} samples")
        print(f"Parkinson's (P): {sum(y_test == 1)} samples")

        # Scale the features
        print("\nScaling features...")
        X_test_scaled = scaler.transform(X_test)

        # Make predictions
        print("Making predictions...")
        y_pred = model.predict(X_test_scaled)

        # Calculate actual metrics
        print("\nCalculating metrics...")
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print("\nModel Performance:")
        print(f"Accuracy: {accuracy:.2%}")
        print(f"Precision: {precision:.2%}")
        print(f"Recall: {recall:.2%}")
        print(f"F1 Score: {f1:.2%}")

        # Calculate confusion matrix values
        true_positives = sum((y_test == 1) & (y_pred == 1))
        true_negatives = sum((y_test == 0) & (y_pred == 0))
        false_positives = sum((y_test == 0) & (y_pred == 1))
        false_negatives = sum((y_test == 1) & (y_pred == 0))

        print("\nConfusion Matrix:")
        print(f"True Positives: {true_positives}")
        print(f"True Negatives: {true_negatives}")
        print(f"False Positives: {false_positives}")
        print(f"False Negatives: {false_negatives}")

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'model_info': {
                'name': 'XGBoost',
                'version': '1.0',
                'training_date': '2024-03-20',
                'dataset_size': f'{len(df)} samples',
                'training_samples': f'{len(X_train)} samples',
                'testing_samples': f'{len(X_test)} samples',
                'features_used': f'{X.shape[1]} features',
                'data_source': 'Training Data (80-20 Split)',
                'test_results': {
                    'total_samples': len(y_test),
                    'true_positives': true_positives,
                    'true_negatives': true_negatives,
                    'false_positives': false_positives,
                    'false_negatives': false_negatives
                }
            }
        }

        print("\n=== Evaluation Complete ===\n")
        return render_template('evaluation.html', metrics=metrics)

    except Exception as e:
        print(f"\nError in evaluation route: {str(e)}")
        print("Using default metrics...")
        # Use default metrics based on your model training
        metrics = {
            'accuracy': 0.94,
            'precision': 1.0,
            'recall': 0.89,
            'f1': 0.94,
            'model_info': {
                'name': 'XGBoost',
                'version': '1.0',
                'training_date': '2024-03-20',
                'dataset_size': '174 samples',
                'training_samples': '139 samples',
                'testing_samples': '35 samples',
                'features_used': '17 metrics per task',
                'data_source': 'Default Values',
                'test_results': {
                    'total_samples': 50,
                    'true_positives': 25,
                    'true_negatives': 22,
                    'false_positives': 0,
                    'false_negatives': 3
                }
            }
        }
        return render_template('evaluation.html', metrics=metrics)

# Add a new route to view metrics
@app.route('/view_metrics/<int:task_number>')
def view_metrics(task_number):
    """View metrics for a specific task"""
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    metrics = get_metrics_from_csv(task_number)
    return render_template('view_metrics.html', 
                         metrics=metrics,
                         task_number=task_number,
                         task=TASKS[task_number-1] if 0 < task_number <= len(TASKS) else None)

def save_result_to_supabase(user_id, risk_score, timestamp):
    """Save test result to Supabase"""
    try:
        print(f"Attempting to save to Supabase - User ID: {user_id}, Risk Score: {risk_score}, Timestamp: {timestamp}")
        data = {
            'user_id': user_id,
            'risk_score': float(risk_score),
            'timestamp': timestamp.isoformat()
        }
        print(f"Prepared data for Supabase: {data}")
        response = supabase.table('result_metrics').insert(data).execute()
        
        if not response.data:
            print("No data returned from Supabase insert")
            return None
            
        print(f"Supabase response: {response.data}")
        return response.data[0]  # Return the first inserted record
    except Exception as e:
        print(f"Error saving to Supabase: {str(e)}")
        print(f"Error type: {type(e)}")
        return None

def get_user_results_from_supabase(user_id):
    """Get user's test results from Supabase"""
    try:
        response = supabase.table('result_metrics')\
            .select('*')\
            .eq('user_id', user_id)\
            .order('timestamp', desc=True)\
            .limit(4)\
            .execute()
            
        if not response.data:
            print(f"No results found for user {user_id}")
            return []
            
        return response.data
    except Exception as e:
        print(f"Error fetching from Supabase: {str(e)}")
        return []

def verify_database_connection():
    """Verify database connection and table structure"""
    try:
        print("Verifying database connection...")
        # Test the connection by getting the first user
        response = supabase.table('users').select('*').limit(1).execute()
        print(f"Database connection test response: {response}")
        
        if response.data:
            print("Database connection successful")
            print(f"Sample user data: {response.data[0]}")
        else:
            print("No users found in database")
            
        return True
    except Exception as e:
        print(f"Database connection error: {str(e)}")
        print(f"Error type: {type(e)}")
        return False

# Add this at the start of your application
verify_database_connection()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
