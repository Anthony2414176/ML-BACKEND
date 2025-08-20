# accounts/views/auth.py
from django.contrib.auth import authenticate
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json

@csrf_exempt
def login_user(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        username = data.get('username')
        password = data.get('password')

        user = authenticate(username=username, password=password)
        if user is not None:
            initials = ''.join([n[0].upper() for n in user.get_full_name().split()]) or user.username[:2].upper()
            return JsonResponse({
                'status': 'success',
                'message': 'Login successful',
                'user': {
                    'username': user.username,
                    'initials': initials,
                    'full_name': user.get_full_name()
                }
            })
        else:
            return JsonResponse({'status': 'fail', 'message': 'Invalid credentials'}, status=401)

    return JsonResponse({'status': 'fail', 'message': 'Only POST allowed'}, status=405)
