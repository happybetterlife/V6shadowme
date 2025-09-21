from fastapi import APIRouter, HTTPException, status

router = APIRouter()


@router.post("/login")
async def login_stub(email: str, password: str):
    # Placeholder implementation until Firebase auth is wired in.
    if not email or not password:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid credentials")
    return {"message": "Stub login successful", "email": email}


@router.post("/logout")
async def logout_stub():
    return {"message": "Stub logout successful"}
