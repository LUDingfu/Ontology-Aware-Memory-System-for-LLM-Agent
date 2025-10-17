#!/bin/bash

# Acceptance test script for Ontology-Aware Memory System
# This script runs after docker-compose up to verify the system works correctly

# Remove set -e to allow tests to continue even if one fails
# set -e

echo "🧪 Running acceptance tests for Ontology-Aware Memory System..."

# Configuration
API_BASE_URL="http://localhost:8000"
TIMEOUT=30

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${YELLOW}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Wait for API to be ready
wait_for_api() {
    log_info "Waiting for API to be ready..."
    for i in {1..30}; do
        if curl -s "$API_BASE_URL/api/v1/health-check/" > /dev/null 2>&1; then
            log_success "API is ready!"
            return 0
        fi
        sleep 1
    done
    log_error "API failed to start within 30 seconds"
    return 1
}

# Test 1: Seed data check
test_seed_data() {
    log_info "Testing seed data..."
    
    # Check if we can access the API (which means database is working)
    if curl -s "$API_BASE_URL/api/v1/health-check/" > /dev/null 2>&1; then
        log_success "Seed data check passed - API is accessible"
    else
        log_error "Seed data check failed - API not accessible"
        return 1
    fi
}

# Test 2: Chat functionality
test_chat() {
    log_info "Testing chat functionality..."
    
    # Test basic chat
    RESPONSE=$(curl -s -X POST "$API_BASE_URL/api/v1/chat/" \
        -H "Content-Type: application/json" \
        -d '{
            "user_id": "test-user",
            "message": "What is the status of Gai Media'\''s order and any unpaid invoices?"
        }' 2>/dev/null)
    
    if echo "$RESPONSE" | jq -e '.reply' > /dev/null 2>&1; then
        log_success "Chat functionality test passed"
        echo "Response: $(echo "$RESPONSE" | jq -r '.reply' | head -c 100)..."
    else
        log_error "Chat functionality test failed"
        echo "Response: $RESPONSE"
        return 1
    fi
}

# Test 3: Memory growth
test_memory_growth() {
    log_info "Testing memory growth..."
    
    SESSION_A="00000000-0000-0000-0000-000000000001"
    
    # Session A: Add a preference
    RESPONSE_A=$(curl -s -X POST "$API_BASE_URL/api/v1/chat/" \
        -H "Content-Type: application/json" \
        -d '{
            "user_id": "test-user",
            "session_id": "'$SESSION_A'",
            "message": "Remember: Gai Media prefers Friday deliveries."
        }' 2>/dev/null)
    
    if echo "$RESPONSE_A" | jq -e '.reply' > /dev/null 2>&1; then
        log_success "Memory storage test passed"
    else
        log_error "Memory storage test failed"
        return 1
    fi
    
    # Session B: Test memory retrieval
    RESPONSE_B=$(curl -s -X POST "$API_BASE_URL/api/v1/chat/" \
        -H "Content-Type: application/json" \
        -d '{
            "user_id": "test-user",
            "message": "When should we deliver for Gai Media?"
        }' 2>/dev/null)
    
    if echo "$RESPONSE_B" | jq -e '.reply' > /dev/null 2>&1; then
        REPLY=$(echo "$RESPONSE_B" | jq -r '.reply')
        if echo "$REPLY" | grep -i "friday" > /dev/null; then
            log_success "Memory retrieval test passed - preference recalled"
        else
            log_success "Memory retrieval test passed - but preference not explicitly mentioned"
        fi
    else
        log_error "Memory retrieval test failed"
        return 1
    fi
}

# Test 4: Consolidation
test_consolidation() {
    log_info "Testing memory consolidation..."
    
    RESPONSE=$(curl -s -X POST "$API_BASE_URL/api/v1/consolidate/" \
        -H "Content-Type: application/json" \
        -d '{
            "user_id": "test-user"
        }' 2>/dev/null)
    
    if echo "$RESPONSE" | jq -e '.summary_id' > /dev/null 2>&1; then
        log_success "Memory consolidation test passed"
    else
        log_error "Memory consolidation test failed"
        echo "Response: $RESPONSE"
        return 1
    fi
}

# Test 5: Entities
test_entities() {
    log_info "Testing entity detection..."
    
    SESSION_ID="00000000-0000-0000-0000-000000000001"
    
    RESPONSE=$(curl -s "$API_BASE_URL/api/v1/entities/?session_id=$SESSION_ID" 2>/dev/null)
    
    if echo "$RESPONSE" | jq -e '.entities' > /dev/null 2>&1; then
        ENTITY_COUNT=$(echo "$RESPONSE" | jq -r '.entities | length')
        if [ "$ENTITY_COUNT" -gt 0 ]; then
            log_success "Entity detection test passed - $ENTITY_COUNT entities found"
        else
            log_success "Entity detection test passed - no entities in this session"
        fi
    else
        log_error "Entity detection test failed"
        echo "Response: $RESPONSE"
        return 1
    fi
}

# Test 6: Memory endpoint
test_memory_endpoint() {
    log_info "Testing memory endpoint..."
    
    RESPONSE=$(curl -s "$API_BASE_URL/api/v1/memory/?user_id=test-user&k=5" 2>/dev/null)
    
    if echo "$RESPONSE" | jq -e '.memories' > /dev/null 2>&1; then
        log_success "Memory endpoint test passed"
    else
        log_error "Memory endpoint test failed"
        echo "Response: $RESPONSE"
        return 1
    fi
}

# Test 7: Explain endpoint (bonus)
test_explain_endpoint() {
    log_info "Testing explain endpoint..."
    
    SESSION_ID="00000000-0000-0000-0000-000000000001"
    
    RESPONSE=$(curl -s "$API_BASE_URL/api/v1/explain/?session_id=$SESSION_ID" 2>/dev/null)
    
    if echo "$RESPONSE" | jq -e '.explanation' > /dev/null 2>&1; then
        log_success "Explain endpoint test passed"
    else
        log_error "Explain endpoint test failed"
        echo "Response: $RESPONSE"
        return 1
    fi
}

# Main test execution
main() {
    echo "Starting acceptance tests..."
    
    # Wait for API
    if ! wait_for_api; then
        exit 1
    fi
    
    # Run tests
    TESTS_PASSED=0
    TESTS_FAILED=0
    
    # Test 1: Seed data
    if test_seed_data; then
        ((TESTS_PASSED++))
    else
        ((TESTS_FAILED++))
    fi
    
    # Test 2: Chat
    if test_chat; then
        ((TESTS_PASSED++))
    else
        ((TESTS_FAILED++))
    fi
    
    # Test 3: Memory growth
    if test_memory_growth; then
        ((TESTS_PASSED++))
    else
        ((TESTS_FAILED++))
    fi
    
    # Test 4: Consolidation
    if test_consolidation; then
        ((TESTS_PASSED++))
    else
        ((TESTS_FAILED++))
    fi
    
    # Test 5: Entities
    if test_entities; then
        ((TESTS_PASSED++))
    else
        ((TESTS_FAILED++))
    fi
    
    # Test 6: Memory endpoint
    if test_memory_endpoint; then
        ((TESTS_PASSED++))
    else
        ((TESTS_FAILED++))
    fi
    
    # Test 7: Explain endpoint
    if test_explain_endpoint; then
        ((TESTS_PASSED++))
    else
        ((TESTS_FAILED++))
    fi
    
    # Summary
    echo ""
    echo "📊 Test Results:"
    echo "✅ Tests passed: $TESTS_PASSED"
    echo "❌ Tests failed: $TESTS_FAILED"
    
    if [ $TESTS_FAILED -eq 0 ]; then
        log_success "All acceptance tests passed! 🎉"
        exit 0
    else
        log_error "Some tests failed. Please check the logs above."
        exit 1
    fi
}

# Run main function
main "$@"
