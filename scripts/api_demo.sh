#!/usr/bin/env bash
# Exercise the API: valid request, missing field, wrong type, invalid business value, health.
# Usage: scripts/api_demo.sh [BASE_URL]   (default http://127.0.0.1:8000)
BASE="${1:-http://127.0.0.1:8000}"
VALID='{"mean_radius":14.1,"mean_texture":19.3,"mean_perimeter":92.0,"mean_area":654.9,"mean_smoothness":0.096,"mean_compactness":0.104,"mean_concavity":0.089,"mean_concave_points":0.049,"mean_symmetry":0.181,"mean_fractal_dimension":0.063}'
MISSING='{"mean_radius":14.1,"mean_texture":19.3}'
WRONGTYPE='{"mean_radius":"large","mean_texture":19.3,"mean_perimeter":92.0,"mean_area":654.9,"mean_smoothness":0.096,"mean_compactness":0.104,"mean_concavity":0.089,"mean_concave_points":0.049,"mean_symmetry":0.181,"mean_fractal_dimension":0.063}'
BUSINESS='{"mean_radius":-5,"mean_texture":19.3,"mean_perimeter":92.0,"mean_area":654.9,"mean_smoothness":1.7,"mean_compactness":0.104,"mean_concavity":0.089,"mean_concave_points":0.049,"mean_symmetry":0.181,"mean_fractal_dimension":0.063}'

call() {  # name method path [json]
  echo "=== $1 ==="; echo "\$ $2 $BASE$3"
  if [ -n "$4" ]; then
    curl -s -w "\nHTTP_STATUS=%{http_code}\n" -X "$2" "$BASE$3" -H "Content-Type: application/json" -d "$4"
  else
    curl -s -w "\nHTTP_STATUS=%{http_code}\n" -X "$2" "$BASE$3"
  fi
  echo
}
call "1. Health check"                GET  /health
call "2. Valid request"               POST /predict "$VALID"
call "3. Missing required field"      POST /predict "$MISSING"
call "4. Wrong data type"             POST /predict "$WRONGTYPE"
call "5. Invalid business value"      POST /predict "$BUSINESS"
call "6. Model info"                  GET  /info
